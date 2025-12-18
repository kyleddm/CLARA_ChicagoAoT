from datetime import datetime
import pytz
import json
from argparse import Namespace
from typing import Dict, List, Tuple, Any, Optional
import pandas as pd
#example timestamp 2020/06/14 00:05:58
# Example timestamp string and timezone
#timestamp_str = '2025/11/21 15:30:00'
timestart_str = '2025/01/01 00:00:00'


def calcSecDiff(endTime:str, startTime:str=timestart_str, timezone_str:str='America/Chicago', verbose=False)->int:
    #timezone_str = 'America/Chicago'
    # Parse the string into a naive datetime object
    naive_datetime = datetime.strptime(endTime, "%Y/%m/%d %H:%M:%S")
    naive_starttime = datetime.strptime(startTime, "%Y/%m/%d %H:%M:%S")
    # Localize the naive datetime to the specific timezone
    timezone = pytz.timezone(timezone_str)
    localized_datetime = timezone.localize(naive_datetime)
    localized_starttime = timezone.localize(naive_starttime)
    # Convert to Unix timestamp
    unix_timestamp = int(localized_datetime.timestamp())
    start_timestamp = int(localized_starttime.timestamp())
    #remove year and calculate seconds since year starts
    yearSeconds=unix_timestamp-start_timestamp
    if verbose:
        print("original end date:",endTime)
        print("Original end Timestamp:", unix_timestamp)
        print("Original Start Timestamp:", start_timestamp)
        print("seconds since year started:", yearSeconds)
    return yearSeconds

def pruneTime(inDate:str, timezone_str='America/Chicago'):
    #strip the year from the given date
    myDate=datetime.strptime(inDate, "%Y/%m/%d %H:%M:%S")
    myYear=str(myDate.year)
    startTime=myYear+'/01/01 00:00:00'
    yearSecs=calcSecDiff(startTime,inDate, timezone_str=timezone_str, verbose=False)
    return yearSecs

def parse_json_args(args):
    args_dict=vars(args)#convert the arguments from the parser into a dictionary to compare values
    args2=None #the new arguments to be returned will live here
    #if you want to use a config file isntead of putting all the args into the command-line
    try:
        conf_file=args.config_path
        with open(conf_file,'r') as fil:
            configs=json.load(fil)
            for key in configs.keys():
                if key in args_dict.keys():
                    args_dict[key]=configs[key]
        args2 = Namespace(**args_dict) #will overwrite original args with values present in the config file.  this should protect against missing items in the config file since it preserves the default args from the parser.
            
    except FileNotFoundError as e:
        print(f'the file speficied by {args.config_path} cannot be found')
        print(e.errno)
    
    
    return args2

def extract_metadata(pattern: Dict[str, Any],args):
    id_keys=[key for key in pattern if 'id' in key.lower()]
    value_keys=[key for key in pattern if 'value' in key.lower()]
    timestamp_keys=[key for key in pattern if ('timestamp' in key.lower() or 'date' in key.lower())]
    label_keys= [key for key in pattern if key not in id_keys and key not in timestamp_keys and key not in value_keys]
    keys={'ids':id_keys,'values':value_keys,'timestamps':timestamp_keys,'labels':label_keys}
    # extract metadata        
    ids=[]
    values=[]
    timestamps=[]
    labels=[]
    data_headers=args.data_headers
    
    for key in id_keys:
        ids.append(pattern.get(key, 'unknown'))
    for key in value_keys:
        values.append(pattern.get(key, 'unknown'))
    for key in timestamp_keys:
        ts=pattern.get(key, 'unknown')
        if ts!='unknown':
            timestamps.append(pruneTime(ts))
    for key in label_keys:
        key2=pattern.get(key, 'unknown')
        if key2 in data_headers.keys():
            key2=data_headers[key2]
        labels.append(key2)

    metadata={'timestamps':timestamps,'ids':ids,'values':values,'labels':labels}
    
    #adding a loader for sensor context here.  This should really be done in a manner where this is a class that can be loaded into a file, but that would require extensive modifications, so we're going to hold off for now.
    sen_meta=pd.read_csv(args.data_units_file)
    subsys=False
    sen=False
    param=False
    for key in label_keys:
        if key in "subsystem":
            subsys=True
            subsystem=pattern.get(key, 'unknown')
        if key in "sensor":
            sen=True
            sensor=pattern.get(key, 'unknown')
        if key in "parameter":
            param=True
            parameter=pattern.get(key, 'unknown')
        if subsys and sen and param:
            #check the list of sensor metadata to determine th eproper units for the sensor
            result=sen_meta[(sen_meta['subsystem']==subsystem)&(sen_meta['sensor']==sensor)&(sen_meta['parameter']==parameter)]
            units=result['hrf_unit']
    
           
    ##Note the units variable is still a series.  We need to read into it.
    #!!!!!this text needs to be edited to expand the sensor types (pres to pressure, temp, to temperature), add readings, and explain that the time is the number of seconds past the start year.!!!!
    text = f"Host {",".join(map(str,ids))} with {",".join(map(str,labels))}, measuring units {units} at time {",".join(map(str,timestamps))} s since start point\n\n"
    return metadata, text, keys

def extractSensorType(pattern: Dict[str, Any]):
    #metadata, text, meta_keys=extract_metadata(pattern, data_headers)
    sensor_groups = {}
    #    'accelerometer': [],
    #    'gyroscope': [],
    #    'magnetometer': [],
    #    'location': [],
    #    'other': []
    #}
    #print(f'TEST~~~PATTERN: {pattern}~~~TEST')
    for key in pattern.keys():
        print(f'pattern key:{key}')
        if key.lower() == 'parameter':
            if pattern[key] in sensor_groups.keys():
                sensor_groups[pattern[key]].append((pattern[key],pattern['value_hrf']))
            else:
                sensor_groups[pattern[key]]=[(pattern[key],pattern['value_hrf'])]
        if key.lower() =='timestamp':
            if pattern[key] in sensor_groups.keys():
                sensor_groups['timestamp'].append(('timestamp',pruneTime(pattern['timestamp'])))
            else:
                sensor_groups['timestamp']=[('timestamp',pruneTime(pattern['timestamp']))]
    return sensor_groups