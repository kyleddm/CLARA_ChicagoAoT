from datetime import datetime
import pytz
#example timestamp 2020/06/14 00:05:58
# Example timestamp string and timezone
timestamp_str = '2025/11/21 15:30:00'
timestart_str = '2025/01/01 00:00:00'



def calcSecDiff(startTime:str,endTime:str, timezone_str='America/Chicago', verbose=False)->int:
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
        print("original date:",timestamp_str)
        print("Original Timestamp:", unix_timestamp)
        print("Original Timestamp:", start_timestamp)
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
    from argparse import Namespace
    args_dict=vars(args)#convert the arguments from the parser into a dictionary to compare values
    args2=None #the new arguments to be returned will live here
    #if you want to use a config file isntead of putting all the args into the command-line
    try:
        configs=json.load(args.config_path)
        for key in configs.keys:
            if key in args_dict.keys():
                args_dict[key]=configs[key]
        args2 = Namespace(**args_dict) #will overwrite original args with values present in the config file.  this should protect against missing items in the config file since it preserves the default args from the parser.
            
    except FileNotFoundError as e:
        print(f'the file speficied by {args.config_path} cannot be found')
        print(e.errno)
    
    
    return args2