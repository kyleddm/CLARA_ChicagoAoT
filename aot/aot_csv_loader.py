import pandas as pd
import numpy as np
import os
import time
from typing import Dict, List, Any, Optional, Tuple
from operator import itemgetter

class AotCSVLoader:
    #cols: timestamp,node_id,subsystem,sensor,parameter,value_hrf
    

    def __init__(self, csv_file_path: str):
        
        self.csv_file_path = csv_file_path
        self.df = None
        self.param_df=None
        #we truncated the data to reduce size, this tells us the meanings of the truncated components
        self.subsystem_def={'cs':'chemsense','ls':'lightsense','ms':'metsense','pt':'plantower'}
        self.param_def={'hum':'humidity','temp':'temperature','pres':'pressure'}
        self.data_headers={**self.subsystem_def, **self.param_def}
        # verify the file exists        
        if not os.path.exists(csv_file_path):
            print(f"Warning: CSV file not found at {csv_file_path}")
        else:
            try:
                # load the file as a pandas dataframe                
                print(f"Loading Sensor data from {csv_file_path}...")
                self.df = pd.read_csv(csv_file_path)
                print(f"Successfully loaded data with {len(self.df)} rows and {len(self.df.columns)} columns\n")
                #print(f"columns in data: {self.df.columns}\n")
            except Exception as e:
                print(f"Error loading CSV file: {e}")
            try:
                self.df.rename(columns={'value_hrf':'value'},inplace=True)
                #print(f'generalized value column: {self.df.columns}\n')
                if 'Unnamed: 0' in list(self.df.columns):
                    self.df=self.df.drop('Unnamed: 0',axis=1)#csv files saved with the index column
                    #print(f'Columns in data: {self.df.columns}\n')
                self.df['subsystem'] = self.df['subsystem'].apply(lambda x: 'chemsense' if x == 'cs' else ('lightsense' if x=='ls' else ('metsense' if x=='ms' else('plantower' if x=='pt' else (x)))))
                self.df['parameter'] = self.df['parameter'].apply(lambda x: 'humidity' if x == 'hum' else ('temperature' if x=='temp' else ('pressure' if x=='pres' else(x))))
                #print(f'expanded subsystem names: {self.df.columns}\n')
                
                print(f"Columns in data: {list(self.df.columns)}\n")
            except Exception as e:
                print(f'Error parsing dataframe: {e}')
        return
    def load_parameter_units(self,sensorFile:str='./input/sensors.csv'):
        if not os.path.exists(sensorFile):
            print(f"Warning: CSV file not found at {sensorFile}")
        else:
            try:
                # load the file as a pandas dataframe                
                print(f"Loading Sensor info from {sensorFile}...")
                self.param_df = pd.read_csv(sensorFile)
                
            except Exception as e:
                print(f"Error loading CSV file: {e}")
        return self.param_def
    def get_column_names(self) -> List[str]:
        
        if self.df is not None:
            return self.df.columns.tolist()
        return []
    
    def get_available_nodes(self) -> List[str]:

        if self.df is not None and 'node_id' in self.df.columns:
            return self.df['node_id'].unique().tolist()
        return []
    
    def get_available_subsystems(self) -> List[str]:

        if self.df is not None and 'subsystem' in self.df.columns:
            return self.df['subsystem'].unique().tolist()
        return []
    def get_available_parameters(self) -> List[str]:

        if self.df is not None and 'parameter' in self.df.columns:
            return self.df['parameter'].unique().tolist()
        return []
    def get_available_sensors(self) -> List[str]:

        if self.df is not None and 'sensor' in self.df.columns:
            return self.df['sensor'].unique().tolist()
        return []
    def load_node_data(self, node: str = None, subsystem: str = None, sensor:str = None, parameter:str = None, max_samples: int = None) -> List[Dict[str, Any]]:

        if self.df is None:
            return []
        
        # apply filters        
        filtered_df = self.df
        
        if node is not None and 'node_id' in self.df.columns:
            filtered_df = filtered_df[filtered_df['node_id'] == node]
        
        if subsystem is not None and 'subsystem' in self.df.columns:
            filtered_df = filtered_df[filtered_df['subsystem'] == subsystem]
        
        if sensor is not None and 'sensor' in self.df.columns:
            filtered_df = filtered_df[filtered_df['sensor'] == sensor]
        
        if parameter is not None and 'parameter' in self.df.columns:
            filtered_df = filtered_df[filtered_df['parameter'] == parameter]
        
        # limit samples if specified        
        if max_samples is not None:
            filtered_df = filtered_df.head(max_samples)
        
        # convert dataframe rows to dictionaries        
        sensor_data_list = []
        
        for _, row in filtered_df.iterrows():
            sensor_data = {}
            
            # add all columns to the dictionary            
            for column in filtered_df.columns:
                value = row[column]
                
                # handle special data types                
                if pd.isna(value):
                    continue
                elif isinstance(value, (np.int64, np.float64)):
                    sensor_data[column] = float(value)
                else:
                    sensor_data[column] = value
            
            ## add timestamp if not already present            
            #if 'timestamp' not in sensor_data:
            #    sensor_data['timestamp'] = time.strftime("%Y-%m-%dT%H:%M:%S")
            
            sensor_data_list.append(sensor_data)
        
        return sensor_data_list

    def filter_data(self, node: str = None, subsystem: str = None, sensor:str = None, parameter:str = None, max_samples: int = None) -> List[Dict[str, Any]]:

        if self.df is None:
            print("There is no data to sample!\n")
            return []
        
        # apply filters        
        filtered_df = self.df
        
        if node is not None:
            if 'node_id' in self.df.columns:
                filtered_df = filtered_df[filtered_df['node_id'] == node]
            else:
                print(f'Did not find {node} in known subsystems')
        if subsystem is not None:
            if 'subsystem' in self.df.columns:
                filtered_df = filtered_df[filtered_df['subsystem'] == subsystem]
            else:
                print(f'Did not find {subsystem} in known subsystems')
        if sensor is not None: 
            if 'sensor' in self.df.columns:
                filtered_df = filtered_df[filtered_df['sensor'] == sensor]
            else:
                print(f'Did not find {sensor} in known sensors')
        if parameter is not None: 
            if 'parameter' in self.df.columns:
                filtered_df = filtered_df[filtered_df['parameter'] == sensor]
            else:
                print(f'Did not find {parameter} in known parameters')
        
        # limit samples if specified        
        if max_samples is not None:
            filtered_df = filtered_df.head(max_samples)
        
        return filtered_df
    
    def identify_parameter_columns(self) -> Dict[str, List[str]]:
        #For Chicago AoT, this function is likely depreciated; the data is unique by ROW, not column.
        if self.df is None:
            return {}
        
        parameter_list=self.get_available_parameters()
        parameter_groups={}
        for param in parameter_list:
            parameter_groups['param']=[]
        #parameter_groups = {
        #    'temperature': [],
        #    'pressure': [],
        #    'humidity': [],
        #    'ir_intensity': [],
        #    'uv_intensity': [],
        #    'visible_light_intensity': [],
        #    'concentration': [],
        #    'intensity':[],
        #    'pm1_atm':[],
        #    'pm1_cf1':[],
        #}
        
        for column in self.df.columns:
            col_lower = column.lower()
            
            # skip metadata columns            
            if column in ['node_id', 'timestamp', 'subsystem']:
                continue
                
            # categorize columns by sensor type-This fails because the ROWS contain the unique information            
            if any(term in col_lower for term in ['temp', 'temperature']):
                parameter_groups['temperature'].append(column)
            elif any(term in col_lower for term in ['pres', 'pressure']):
                parameter_groups['pressure'].append(column)
            elif any(term in col_lower for term in ['hum', 'humidity']):
                parameter_groups['humidity'].append(column)
            elif any(term in col_lower for term in ['ir','ir_intensity']):
                parameter_groups['ir_intensity'].append(column)
            elif any(term in col_lower for term in ['uv','uv_intensity']):
                parameter_groups['uv_intensity'].append(column)
            elif any(term in col_lower for term in ['light', 'light_intensity', 'visible_light_intensity']):
                parameter_groups['visible_light_intensity'].append(column)
            elif any(term in col_lower for term in ['concentration']):
                parameter_groups['concentration'].append(column)
            elif any(term in col_lower for term in ['intensity']):
                parameter_groups['intensity'].append(column)
            else:
                parameter_groups['other'].append(column)
        
        # remove empty groups        
        return {k: v for k, v in parameter_groups.items() if v}
    
    def get_data_statistics(self) -> Dict[str, Any]:

        if self.df is None:
            return {'error': 'No data loaded'}
        subsystems=self.get_available_subsystems()
        stats = {
            'total_samples': len(self.df),
            'columns': len(self.df.columns),
            'nodes': len(self.get_available_nodes()),
            'total_subsystems': len(subsystems),
            'subsystem_groups':subsystems,
            'parameter_groups': self.get_available_parameters(),
            'sensor_groups': self.get_available_sensors(),
            'missing_values': self.df.isna().sum().sum(),
            'missing_percentage': (self.df.isna().sum().sum() / (self.df.shape[0] * self.df.shape[1])) * 100
        }
        
        return stats
    
    def generate_synthetic_data(self, num_samples: int = 10) -> List[Dict[str, Any]]:
        #for AOT data, this fails because a column can have any type of data in it.  in order to do this effectively, I would need to make each sensor have it's own column, then merge the raw and real values together somehow.  Probably a good idea to drop the value_raw column since it doesn't mesh with the reported sensor ranges
        #cols: timestamp,node_id,subsystem,sensor,parameter,value_raw,value_hrf
        comparison_df = None
        df1 = None
        df2 = None
        df3 = None
        data = []
        
        df1=self.filter_data(node=None, subsystem='metsense', sensor='tsys01', parameter='temperature', max_samples=None)#since data is unique by ROW, we need to sample from rows to get these values.  This is a test case only! 2025NOV21
        #print(f'DF1!!!:{df1}\n')
        df2=self.filter_data(node=None, subsystem='metsense', sensor='htu21d', parameter='humidity', max_samples=None)
        df3=self.filter_data(node=None, subsystem='chemsense', sensor='lps25h', parameter='pressure', max_samples=None)
        comparison_df = df1 + df2 + df3 #pd.concat([df1, df2, df3], ignore_index=True)
        #try: Don;t need to sort.  only usinf this list to make sure we got samples.
            #comparison_df=comparison_df.sort_values(by='timestamp', ascending=True).reset_index(drop=True)
        #    comparison_df = sorted(comparison_df, key=itemgetter("timestamp"))
        #except KeyError as e:
        #    print(f"Error: Missing key {e} in one of the dictionaries.")
        df1=pd.DataFrame(df1)
        df2=pd.DataFrame(df2)
        df3=pd.DataFrame(df3)
        # determine available sensor columns from real data        
        #sensor_groups = self.identify_sensor_columns()
        
        # if we have real data columns, use them as a template        
        if comparison_df is not None:
            if df1 is not None:
                mean_val1 = df1['value'].mean()
                std_val1 = df1['value'].std()
            if df2 is not None:
                mean_val2 = df2['value'].mean()
                std_val2 = df2['value'].std()
            if df3 is not None:
                mean_val3 = df3['value'].mean()
                std_val3 = df3['value'].std()
            # get column names from the first few sensor groups            
            #template_columns = []
            #for group in list(sensor_groups.values())[:3]:
            #    template_columns.extend(group[:3])  # take up to 3 columns from each group            
            # create normal patterns            
            for i in range(num_samples - 1):
                #sample = {
                #    "node_id": "synthetic_host",
                #    "subsystem": "metsemse",
                #    "sensor":"tsys01",
                #    "parameter":"temperature",
                #    "timestamp": f"2023-01-01T{12+i:02d}:00:00"
                #}
                # generate value using distribution from real data                        
                #if not pd.isna(mean_val) and not pd.isna(std_val) and std_val > 0:
                #    sample['value'] = mean_val + np.random.normal(0, std_val * 0.5)
                #else:
                #    sample['value'] = np.random.normal(0, 1)
                
                #data.append(sample)
                switcher=i%3
                match switcher:
                    case 0:
                        if not pd.isna(mean_val1) and not pd.isna(std_val1) and std_val1 > 0:
                            val = mean_val1 + np.random.normal(0, std_val1 * 0.5)
                        else:
                            val = np.random.uniform(-5, 5)
                        data.append({
                            "node_id": "synthetic_host",
                            "subsystem": "metsense",
                            "sensor":"tsys01",
                            "parameter":"temperature",
                            "timestamp": f"2023-01-01T{12+i:02d}:00:00",
                            "value": val
                        })
                    case 1:
                        if not pd.isna(mean_val3) and not pd.isna(std_val3) and std_val1 > 0:
                            val = mean_val3 + np.random.normal(0, std_val3 * 0.5)
                        else:
                            val = np.random.uniform(1000, 3000)
                        data.append({
                            "node_id": "synthetic_host",
                            "subsystem": "chemsense",
                            "sensor":"lps25h",
                            "parameter":"pressure",
                            "timestamp": f"2023-01-01T{12+i:02d}:00:00",
                            "value": val
                        })
                    case 2:
                        if not pd.isna(mean_val2) and not pd.isna(std_val2) and std_val2 > 0:
                            val = mean_val2 + np.random.normal(0, std_val2 * 0.5)
                        else:
                            val = np.random.uniform(40, 50)
                        data.append({
                            "node_id": "synthetic_host",
                            "subsystem": "metsense",
                            "sensor":"htu21d",
                            "parameter":"humidity",
                            "timestamp": f"2023-01-01T{12+i:02d}:00:00",
                            "value": val
                        })                    
                
            
            # create an anomalous pattern with significant deviations            
            anomaly = {
                "node_id": "synthetic_host",
                "subsystem": "metsense",
                "sensor":"tsys01",
                "parameter":"temperature",
                "timestamp": "2023-01-01T22:30:00"
            }  
            # generate anomalous value - much higher deviation                    
            #if not pd.isna(mean_val) and not pd.isna(std_val) and std_val > 0:
            #    # 50% chance of higher or lower anomaly                        
            #    if np.random.random() > 0.5:
            #        anomaly['value'] = mean_val + np.random.normal(0, std_val * 3)
            #    else:
            #        anomaly['value'] = mean_val - np.random.normal(0, std_val * 3)
            #else:
            #    anomaly['value'] = np.random.normal(0, 20)
            #data.append(anomaly)
            picker=np.random.uniform(0,2)
            match picker:
                case 0:            
                    if not pd.isna(mean_val1) and not pd.isna(std_val1) and std_val1 > 0:
                    # 50% chance of higher or lower anomaly                        
                        if np.random.random() > 0.5:
                            val = mean_val1 + np.random.normal(0, std_val1 * 3)
                        else:
                            val = mean_val1 - np.random.normal(0, std_val1 * 3)
                    else:
                        val = np.random.uniform(40, 50)
                    data.append({
                        "node_id": "synthetic_host",
                        "subsystem": "metsense",
                        "sensor":"tsys01",
                        "parameter":"temperature",
                        "timestamp": "2023-01-01T22:30:00",
                        "value": val
                    })
                case 1:
                    if not pd.isna(mean_val3) and not pd.isna(std_val3) and std_val3 > 0:
                    # 50% chance of higher or lower anomaly                        
                        if np.random.random() > 0.5:
                            val = mean_val3 + np.random.normal(0, std_val3 * 3)
                        else:
                            val = mean_val3 - np.random.normal(0, std_val3 * 3)
                    else:
                        val = np.random.uniform(0, 900)
                    data.append({
                        "node_id": "synthetic_host",
                        "subsystem": "chemsense",
                        "sensor":"lps25h",
                        "parameter":"pressure",
                        "timestamp": "2023-01-01T22:30:00",
                        "value": val
                    })
                case 2:
                    if not pd.isna(mean_val2) and not pd.isna(std_val2) and std_val2 > 0:
                    # 50% chance of higher or lower anomaly                        
                        if np.random.random() > 0.5:
                            val = mean_val2 + np.random.normal(0, std_val2 * 3)
                        else:
                            val = mean_val2 - np.random.normal(0, std_val2 * 3)
                    else:
                        val = np.random.uniform(60, 100)
                    data.append({
                        "node_id": "synthetic_host",
                        "subsystem": "metsense",
                        "sensor":"htu21d",
                        "parameter":"humidity",
                        "timestamp": "2023-01-01T22:30:00",
                        "value": val
                    })
        else:
            # fall back to basic synthetic data if no real data is available            
            # create normal temp patterns           
            for i in range(num_samples - 1):
                switcher=i%3
                match switcher:
                    case 0:
                        data.append({
                            "node_id": "synthetic_host",
                            "subsystem": "metsense",
                            "sensor":"tsys01",
                            "parameter":"temperature",
                            "timestamp": f"2023-01-01T{12+i:02d}:00:00",
                            "value": 0 + np.random.uniform(0, 2)
                        })
                    case 1:
                        data.append({
                            "node_id": "synthetic_host",
                            "subsystem": "chemsense",
                            "sensor":"lps25h",
                            "parameter":"pressure",
                            "timestamp": f"2023-01-01T{12+i:02d}:00:00",
                            "value": 0 + np.random.uniform(1000, 3000)
                        })
                    case 2:
                        data.append({
                            "node_id": "synthetic_host",
                            "subsystem": "metsense",
                            "sensor":"htu21d",
                            "parameter":"humidity",
                            "timestamp": f"2023-01-01T{12+i:02d}:00:00",
                            "value": 0 + np.random.uniform(40, 50)
                        })
            
            # create an anomalous pattern
            picker=np.random.uniform(0,2)
            match picker:
                case 0:            
                    data.append({
                        "node_id": "synthetic_host",
                        "subsystem": "metsense",
                        "sensor":"tsys01",
                        "parameter":"temperature",
                        "timestamp": "2023-01-01T22:30:00",
                        "value": 0 + np.random.uniform(10, 30)
                    })
                case 1:
                    data.append({
                        "node_id": "synthetic_host",
                        "subsystem": "chemsense",
                        "sensor":"lps25h",
                        "parameter":"pressure",
                        "timestamp": "2023-01-01T22:30:00",
                        "value": 0 + np.random.uniform(0, 900)
                    })
                case 2:
                    data.append({
                        "node_id": "synthetic_host",
                        "subsystem": "metsense",
                        "sensor":"htu21d",
                        "parameter":"humidity",
                        "timestamp": "2023-01-01T22:30:00",
                        "value": 0 + np.random.uniform(60, 100)
                    })
        
        return data


if __name__ == "__main__":
    # test loading from csv file    
    csv_path = "./input/big2018-03-30_00.36.46.csv"
    loader = AotCSVLoader(csv_path)
    
    # print statistics    
    stats = loader.get_data_statistics()
    print("\nDataset Statistics:")
    for key, value in stats.items():
        if key == 'sensor_groups':
            print(f"Sensor Groups:")
            for group, columns in value.items():
                print(f"  - {group}: {len(columns)} columns")
        else:
            print(f"  - {key}: {value}")
    
    # print available users and activities    
    nodes = loader.get_available_nodes()
    subsystems = loader.get_available_subsystems()
    sensors = loader.get_available_sensors()
    parameters = loader.get_available_parameters()
    
    print(f"\nFound {len(nodes)} nodes")
    if nodes:
        print(f"Sample users: {nodes[:5]}")
    print(f"Found {len(subsystems)} subsystems")
    if subsystems:
        print(f"Sample usbsystems: {subsystems[:10]}")
    print(f"Found {len(sensors)} sensors")
    if sensors:
        print(f"Sample sensors: {sensors[:10]}")
    print(f"Found {len(parameters)} parameters")
    if parameters:
        print(f"Sample parameters: {parameters[:5]}")
    
    
    # load some sample data    
    if nodes and subsystems and sensors and parameters:
        node = nodes[0]
        subsystem = subsystems[0]
        sensor= sensors[0]
        parameter=parameters[0]
        samples = loader.load_user_data(node, subsystem, sensor, parameter, max_samples=5)
        
        print(f"\nSample data for node {node}, subsystem {subsystem}, sensor {sensor}, parameter {parameter}:")
        for i, sample in enumerate(samples):
            print(f"Sample {i+1}: {list(sample.keys())[:10]}")
            
            # print a few sensor values            
            sensor_values = {k: v for k, v in sample.items() 
                            if k not in ['node_id', 'subsystem', 'sensor', 'parameter', 'timestamp'] and i < 3}
            print(f"  Sensor values (sample): {list(sensor_values.items())[:5]}")
    
    # test synthetic data generation    
    print("\nGenerating synthetic data based on the dataset patterns...")
    synthetic_data = loader.generate_synthetic_data(5)
    print(f"Generated {len(synthetic_data)} synthetic samples")
    if synthetic_data:
        print(f"Sample synthetic data: {list(synthetic_data[0].keys())[:10]}")