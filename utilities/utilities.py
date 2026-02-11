from datetime import datetime
import pytz
import json
from argparse import Namespace
from typing import Dict, List, Tuple, Any, Optional
import pandas as pd
#from . import config  # relative import
#example timestamp 2020/06/14 00:05:58
# Example timestamp string and timezone
#timestamp_str = '2025/11/21 15:30:00'
DEFAULT_CONFIG='./config.json'
timestart_str = '2025/01/01 00:00:00'
def load_config(config_file:str=DEFAULT_CONFIG):
    conf=''
    with open(config_file,'r') as cf:
        conf=json.load(cf)
    return conf


config=load_config(DEFAULT_CONFIG)

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
#    #strip the year from the given date
#    myDate=datetime.strptime(inDate, "%Y/%m/%d %H:%M:%S")
#    myYear=str(myDate.year)
#    startTime=myYear+'/01/01 00:00:00'
#    yearSecs=calcSecDiff(inDate,startTime, timezone_str=timezone_str, verbose=False)
#    return yearSecs
    return

def returnUnixTime(date:str, syntax:str="%Y/%m/%d %H:%M:%S"):
    return datetime.strptime(date, syntax).timestamp()

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
    #Note: need a concise way to prune the timestamp at time of pattern storage and sample check.
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
    #print(f'DATA HEADERS!!!: {data_headers}')
    for key in id_keys:
        ids.append(pattern.get(key, 'unknown'))
    for key in value_keys:
        values.append(pattern.get(key, 'unknown'))
    for key in timestamp_keys:
        ts=pattern.get(key, 'unknown')
        if ts!='unknown':
            timestamps.append(returnUnixTime(ts))
    for key in label_keys:
        key2=pattern.get(key, 'unknown')
        #print(f'key: {key}, key2: {key2}, data headers keys: {data_headers.keys()}')
        if key2 in data_headers.keys():
            key2=data_headers[key2]
        labels.append(key2)

    metadata={'timestamps':timestamps,'ids':ids,'values':values,'labels':labels}
    #print(f'CONIFG:{config}')
    headers=config['data_headers']
    #adding a loader for sensor context here.  This should really be done in a manner where this is a class that can be loaded into a file, but that would require extensive modifications, so we're going to hold off for now.
    sen_meta=pd.read_csv(args.data_units_file)
    subsys=False
    sen=False
    param=False
    units=None
    #print(f'LABEL KEYS:{label_keys}')
    #print(f'SEN_META_KEYS{sen_meta.keys()}')
    for key in label_keys:
        val=pattern.get(key, 'unknown')
        if key in "subsystem":
            subsys=True
            if val in headers.keys():
                subsystem=headers[val]
            else:
                subsystem=val
        if key in "sensor":
            sen=True
            if val in headers.keys():
                sensor=headers[val]
            else:
                sensor=val
        if key in "parameter":
            param=True
            if val in headers.keys():
                parameter=headers[val]
            else:
                parameter=val
        if subsys and sen and param:
            #check the list of sensor metadata to determine the proper units for the sensor
            #print(f'SUBSYSTEM:{subsystem} SENSOR:{sensor} PARAMETER:{parameter}')
            result=sen_meta[(sen_meta['subsystem']==subsystem)&(sen_meta['sensor']==sensor)&(sen_meta['parameter']==parameter)]
            #print(f'RESULT:{result}')
            units=result['hrf_unit']#.iloc[1]
    
           
    ##Note the units variable is still a series.  We need to read into it.
    #!!!!!this text needs to be edited to expand the sensor types (pres to pressure, temp, to temperature), add readings, and explain that the time is the number of seconds past the start year.!!!!
    
    text = f"Host {",".join(map(str,ids))} with {",".join(map(str,labels))}, measuring units {units} at time {",".join(map(str,timestamps))} \n\n"
    return metadata, text, keys

def provideGuidance(keys:list[str]):
        
        guidance= 'You are seeing sensor data from Chicago, IL USA. Keep this in mind when determining if something is anomalous or not.\n'
        guidance += 'Chicago is not typically an extreme weather location, but can be affected by lake effects, polar vortices in winter, and humid conditions in summer.'
        if 'timestamp' in keys:
            guidance += 'The sensor data is heavily correllated with the date.  This date is provided as a feature and is in the form of UNIX Time.\n'
        if 'temperature' in keys:
            guidance += '\nFor temperature data:\n'
            guidance += '-temperature ranges on earth are generally normal between 0C in winter to 30C in summer generally.\n'
        if 'pressure' in keys:
            guidance += '\nFor Pressure Data:\n'
            guidance += '-pressure ranges variy with weather and altitude, but a low pressure is around 1000 hPa at sea level.\n'
            guidance += '-High pressure systems might show numbers at or above 3000 hPa at sea level.\n'
        if 'humidity' in keys:
            guidance += '\nFor Humidity Data\n'
            guidance += '-humidity can only be within the range of 0 to 100 \% relative humitity.\n'
            guidance += '-humidity is usually lower during colder periods and higher during warmer periods, but closeness to a body of water or a damp environment like a swamp can effect this.\n'
            
        ## add sensor-specific guidance        
        #if 'acc_x' in sensor_data and 'acc_y' in sensor_data and 'acc_z' in sensor_data:
        #    guidance += "\nFor accelerometer data:\n"
        #    guidance += "- Normal walking typically shows y-axis values around 9.8 (gravity)\n"
        #    guidance += "- Sudden changes in x or z axis may indicate unusual movements\n"
        
        #if 'gyro_x' in sensor_data and 'gyro_y' in sensor_data and 'gyro_z' in sensor_data:
        #    guidance += "\nFor gyroscope data:\n"
        #    guidance += "- High values indicate rapid rotation\n"
        #    guidance += "- Normal activities have consistent patterns in gyroscope readings\n"
        return guidance

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
        #print(f'pattern key:{key}')
        if key.lower() == 'parameter':
            if pattern[key] in sensor_groups.keys():
                sensor_groups[pattern[key]].append((pattern[key],pattern['value']))
            else:
                sensor_groups[pattern[key]]=[(pattern[key],pattern['value'])]
        if key.lower() =='timestamp':
            #Note: need a concise way to prune the timestamp at time of pattern storage and sample check.
            if pattern[key] in sensor_groups.keys():
                sensor_groups['timestamp'].append(('timestamp',pattern['timestamp']))
            else:
                sensor_groups['timestamp']=[('timestamp',pattern['timestamp'])]
    return sensor_groups

def closest_divisor(n: int, target: int, mode: str = "any") -> int:
    """
    Find the closest divisor of n to the target value.

    Args:
        n (int): The number whose divisors are considered.
        target (int): The value to compare divisors against.
        mode (str): "any" for closest overall,
                    "smaller" for closest <= target,
                    "larger" for closest >= target.

    Returns:
        int: The closest divisor to the target based on mode.
             Returns None if no valid divisor found.
             
    This function was generated by Microsoft CoPilot, 2026JAN30
    """
    if n == 0:
        raise ValueError("Zero has infinitely many divisors; not supported.")
    if target <= 0:
        raise ValueError("Target must be a positive integer.")

    # Get all positive divisors of n
    divisors = set()
    for i in range(1, int(abs(n) ** 0.5) + 1):
        if n % i == 0:
            divisors.add(i)
            divisors.add(abs(n) // i)

    # Filter based on mode
    if mode == "smaller":
        divisors = [d for d in divisors if d <= target]
    elif mode == "larger":
        divisors = [d for d in divisors if d >= target]
    elif mode != "any":
        raise ValueError("Mode must be 'any', 'smaller', or 'larger'.")

    if not divisors:
        return None

    # Find divisor with smallest absolute difference to target
    return min(divisors, key=lambda d: abs(d - target))