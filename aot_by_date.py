#package importing
#from pyspark import ml
from pyspark.sql.functions import *
#from spark_rapids_ml.clustering import KMeans
from cuml.cluster import HDBSCAN
from pyspark.sql.functions import concat_ws,col,lit, row_number, monotonically_increasing_id
from pyspark.sql.window import Window
from pyspark.sql import SparkSession
import pickle as pkl
import pandas as pd
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.feature import Normalizer
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
#from lenspy import DynamicPlot
import pickle
from datetime import datetime
from pyspark.sql.types import FloatType
#from cuml import DBSCAN as dbscan
from sklearn import preprocessing
from numpy import int64
#from sklearn.metrics import davies_bouldin_score
import findspark
import json
def calcUnixTime(timestamp:str):
    from datetime import datetime
    import pytz
    #code from Bing 2025OCT116
    # Define the date, time, and timezone
    date_str = timestamp#"2025-10-16 15:30:00"  # Example date and time
    timezone_str = "UTC"#"America/New_York"  # Example timezone
    # Parse the date and time
    dt = datetime.strptime(date_str, "%Y/%m/%d %H:%M:%S")
    # Localize the datetime object to the specified timezone
    timezone = pytz.timezone(timezone_str)
    localized_dt = timezone.localize(dt)
    # Convert to UNIX timestamp
    unix_timestamp = int(localized_dt.timestamp())
    print("UNIX Timestamp:", unix_timestamp)
    return unix_timestamp
#https://spark.apache.org/docs/1.2.2/ml-guide.html
#https://pyshark.com/davies-bouldin-index-for-k-means-clustering-evaluation-in-python/
#import torch
findspark.init()
#export PYSPARK_SUBMIT_ARGS="--master local[2] pyspark-shell"
#export JAVA_HOME="/usr/bin/java"
AOT_ROOT='/mnt/e/datasets/Chicago-AoT-dataset/AoT_Chicago.complete.2022-08-31/'
OUT_ROOT='/mnt/e/outputs/chicago_aot/'
#pull in dataset(s) and start spark session
#with open('pre-processed/AoT_dataset_0.pkl', 'rb') as fil:
#    pdf=pd.DataFrame(pkl.load(fil))
#spark=SparkSession.builder.getOrCreate()
spark = SparkSession.builder.master('local[*]').config("spark.driver.memory", "30g").config("spark.driver.maxResultSize","20g").appName('Chicago_AoT').getOrCreate()
spark.conf.set("spark.sql.session.timeZone", "UTC")
#spark = SparkSession.builder.config("spark.driver.memory", "20g").getOrCreate()
spark.conf.set('spark.rapids.sql.enabled','true')
spark.conf.set('spark.rapids.memory.gpu.pooling.enabled','true')
stripdf=spark.read.csv(AOT_ROOT+'data.csv',sep=',', header=True)
df_nodes=spark.read.csv(AOT_ROOT+'nodes.csv',sep=',', header=True)
df_sensors=spark.read.csv(AOT_ROOT+'sensors.csv',sep=',', header=True)
#add Lat/Lon to the main dataset
stripdf = stripdf.withColumn('index', monotonically_increasing_id()).join(df_nodes, stripdf.node_id == df_nodes.node_id, 'left').orderBy('index').select('timestamp', stripdf.node_id, 'subsystem', 'sensor', 'parameter', 'value_raw', 'value_hrf', 'lat', 'lon')
max_id_substring='001e061' #this is the portion of the node_id that is identical to every node
strip_sens=['chemsense', 'metsense', 'loadavg', 'mem', 'time', 'device', 'net_rx', 'net_tx', 'ping', 'media', 'modem', 'disk_used', 'disk_size', 'disk_used_ratio', 'service_active', 'plugins', 'wagman_fc', 'wagman_cu', 'wagman_enabled', 'wagman_vdc', 'wagman_hb', 'wagman_stopping', 'wagman_starting', 'wagman_killing', 'wagman_th', 'wagman_comm', 'wagman_uptime', 'image_detector']
sensor_name_truncation={'chemsense':'cs','alphasense':'as','metsense':'ms','plantower':'pt','audio':'aud','lightsense':'ls','wagman':'wg','microphone':'mic','image':'img'}
#truncate all names of subsystems and node ids for space
stripdf=stripdf.filter(~col('sensor').isin(strip_sens))
stripdf=stripdf.withColumn("subsystem", when(col("subsystem") == "chemsense", 'cs').otherwise(col("subsystem")))
stripdf=stripdf.withColumn("subsystem", when(col("subsystem") == "metsense", 'ms').otherwise(col("subsystem")))
stripdf=stripdf.withColumn("subsystem", when(col("subsystem") == "lightsense", 'ls').otherwise(col("subsystem")))
stripdf=stripdf.withColumn("subsystem", when(col("subsystem") == "alphasense", 'as').otherwise(col("subsystem")))
stripdf=stripdf.withColumn("subsystem", when(col("subsystem") == "plantower", 'pt').otherwise(col("subsystem")))
stripdf=stripdf.withColumn("subsystem", when(col("subsystem") == "audio", 'aud').otherwise(col("subsystem")))
stripdf=stripdf.withColumn("subsystem", when(col("subsystem") == "wagman", 'wg').otherwise(col("subsystem")))
stripdf=stripdf.withColumn("subsystem", when(col("subsystem") == "microphone", 'mic').otherwise(col("subsystem")))
stripdf=stripdf.withColumn("subsystem", when(col("subsystem") == "image", 'img').otherwise(col("subsystem")))
stripdf=stripdf.withColumn("node_id",expr("substring(node_id, 8, 12)"))
#add UNIX timestamp instead of timestamp
stripdf = stripdf.withColumn('unixTime',unix_timestamp(to_timestamp('timestamp','yyyy/MM/dd HH:mm:ss')))
stripdf=stripdf.drop('timestamp')
print(spark.conf.get('spark.driver.maxResultSize'))
#time_start=calcUnixTime('2018/01/01 00:00:00')
#time_end=calcUnixTime('2018/01/15 23:59:59')
with open('dates.json','r') as dateFil:
    timelist=json.load(dateFil)
    for dates in timelist:
        time_start=calcUnixTime(dates['time_start'])
        time_end=calcUnixTime(dates['time_end'])
        test=stripdf.filter((stripdf.unixTime >= time_start)& (stripdf.unixTime<=time_end))#.contains('2018/01/01'))
        testdf=test.toPandas()
        testdf.to_csv(OUT_ROOT+dates['title']+'.csv')
