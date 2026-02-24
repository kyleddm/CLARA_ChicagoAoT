import pandas as pd
import numpy as np
import os
import time
from typing import Dict, List, Any, Optional, Tuple

class ExtraSensoryCSVLoader:

    def __init__(self, csv_file_path: str):
        
        self.csv_file_path = csv_file_path
        self.df = None
        
        # verify the file exists        
        if not os.path.exists(csv_file_path):
            print(f"Warning: CSV file not found at {csv_file_path}")
        else:
            try:
                # load the file as a pandas dataframe                
                print(f"Loading ExtraSensory data from {csv_file_path}...")
                self.df = pd.read_csv(csv_file_path)
                print(f"Successfully loaded data with {len(self.df)} rows and {len(self.df.columns)} columns")
            except Exception as e:
                print(f"Error loading CSV file: {e}")
    
    def get_column_names(self) -> List[str]:
        
        if self.df is not None:
            return self.df.columns.tolist()
        return []
    
    def get_available_users(self) -> List[str]:

        if self.df is not None and 'hostID' in self.df.columns:
            return self.df['hostID'].unique().tolist()
        return []
    
    def get_available_activities(self) -> List[str]:

        if self.df is not None and 'activity' in self.df.columns:
            return self.df['activity'].unique().tolist()
        return []
    
    def load_user_data(self, user_id: str = None, activity: str = None, max_samples: int = None) -> List[Dict[str, Any]]:

        if self.df is None:
            return []
        
        # apply filters        
        filtered_df = self.df
        
        if user_id is not None and 'hostID' in self.df.columns:
            filtered_df = filtered_df[filtered_df['hostID'] == user_id]
        
        if activity is not None and 'activity' in self.df.columns:
            filtered_df = filtered_df[filtered_df['activity'] == activity]
        
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
            
            # add timestamp if not already present            
            if 'timestamp' not in sensor_data:
                sensor_data['timestamp'] = time.strftime("%Y-%m-%dT%H:%M:%S")
            
            sensor_data_list.append(sensor_data)
        
        return sensor_data_list
    
    def identify_sensor_columns(self) -> Dict[str, List[str]]:

        if self.df is None:
            return {}
        
        sensor_groups = {
            'accelerometer': [],
            'gyroscope': [],
            'magnetic': [],
            'location': [],
            'audio': [],
            'phone_state': [],
            'other': []
        }
        
        for column in self.df.columns:
            col_lower = column.lower()
            
            # skip metadata columns            
            if column in ['user_id', 'timestamp', 'activity', 'uuid']:
                continue
                
            # categorize columns by sensor type            
            if any(term in col_lower for term in ['acc', 'accel']):
                sensor_groups['accelerometer'].append(column)
            elif any(term in col_lower for term in ['gyro', 'rotation']):
                sensor_groups['gyroscope'].append(column)
            elif any(term in col_lower for term in ['mag', 'magnetic']):
                sensor_groups['magnetic'].append(column)
            elif any(term in col_lower for term in ['loc', 'gps', 'lat', 'lng']):
                sensor_groups['location'].append(column)
            elif any(term in col_lower for term in ['audio', 'sound', 'mic']):
                sensor_groups['audio'].append(column)
            elif any(term in col_lower for term in ['phone', 'call', 'screen']):
                sensor_groups['phone_state'].append(column)
            else:
                sensor_groups['other'].append(column)
        
        # remove empty groups        
        return {k: v for k, v in sensor_groups.items() if v}
    
    def get_data_statistics(self) -> Dict[str, Any]:

        if self.df is None:
            return {'error': 'No data loaded'}
        
        stats = {
            'total_samples': len(self.df),
            'columns': len(self.df.columns),
            'users': len(self.get_available_users()),
            'activities': len(self.get_available_activities()),
            'sensor_groups': self.identify_sensor_columns(),
            'missing_values': self.df.isna().sum().sum(),
            'missing_percentage': (self.df.isna().sum().sum() / (self.df.shape[0] * self.df.shape[1])) * 100
        }
        
        return stats
    
    def generate_synthetic_data(self, num_samples: int = 10) -> List[Dict[str, Any]]:

        data = []
        
        # determine available sensor columns from real data        
        sensor_groups = self.identify_sensor_columns()
        
        # if we have real data columns, use them as a template        
        if sensor_groups and self.df is not None:
            # get column names from the first few sensor groups            
            template_columns = []
            for group in list(sensor_groups.values())[:3]:
                template_columns.extend(group[:3])  # take up to 3 columns from each group            
            # create normal patterns            
            for i in range(num_samples - 1):
                sample = {
                    "hostID": "synthetic_host",
                    "activity": "walking",
                    "timestamp": f"2023-01-01T{12+i:02d}:00:00"
                }
                
                # add values based on real column ranges                
                for col in template_columns:
                    if col in self.df.columns:
                        # calculate mean and std for the column                        
                        mean_val = self.df[col].mean()
                        std_val = self.df[col].std()
                        
                        # generate value using distribution from real data                        
                        if not pd.isna(mean_val) and not pd.isna(std_val) and std_val > 0:
                            sample[col] = mean_val + np.random.normal(0, std_val * 0.5)
                        else:
                            sample[col] = np.random.normal(0, 1)
                
                data.append(sample)
            
            # create an anomalous pattern with significant deviations            
            anomaly = {
                "hostID": "synthetic_host",
                "activity": "walking",
                "timestamp": "2023-01-01T22:30:00"
            }
            
            for col in template_columns:
                if col in self.df.columns:
                    # calculate mean and std for the column                    
                    mean_val = self.df[col].mean()
                    std_val = self.df[col].std()
                    
                    # generate anomalous value - much higher deviation                    
                    if not pd.isna(mean_val) and not pd.isna(std_val) and std_val > 0:
                        # 50% chance of higher or lower anomaly                        
                        if np.random.random() > 0.5:
                            anomaly[col] = mean_val + np.random.normal(0, std_val * 3)
                        else:
                            anomaly[col] = mean_val - np.random.normal(0, std_val * 3)
                    else:
                        anomaly[col] = np.random.normal(0, 5)
            
            data.append(anomaly)
            
        else:
            # fall back to basic synthetic data if no real data is available            
            # create normal walking patterns           
            for i in range(num_samples - 1):
                data.append({
                    "hostID": "synthetic_host",
                    "activity": "walking",
                    "timestamp": f"2023-01-01T{12+i:02d}:00:00",
                    "acc_x": 0.1 + np.random.normal(0, 0.05),
                    "acc_y": 9.8 + np.random.normal(0, 0.1),
                    "acc_z": 0.2 + np.random.normal(0, 0.05),
                    "gyro_x": 0.01 + np.random.normal(0, 0.01),
                    "gyro_y": 0.02 + np.random.normal(0, 0.01),
                    "gyro_z": 0.01 + np.random.normal(0, 0.01)
                })
            
            # create an anomalous walking pattern            
            data.append({
                "hostID": "synthetic_host",
                "activity": "walking",
                "timestamp": "2023-01-01T22:30:00",
                "acc_x": 0.5 + np.random.normal(0, 0.05),
                "acc_y": 8.5 + np.random.normal(0, 0.1),
                "acc_z": 0.8 + np.random.normal(0, 0.05),
                "gyro_x": 0.1 + np.random.normal(0, 0.01),
                "gyro_y": 0.2 + np.random.normal(0, 0.01),
                "gyro_z": 0.1 + np.random.normal(0, 0.01)
            })
        
        return data


if __name__ == "__main__":
    # test loading from csv file    
    csv_path = "/home/ai-lab2/GAIN-Pytorch-master/data/combined.csv"
    loader = ExtraSensoryCSVLoader(csv_path)
    
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
    users = loader.get_available_users()
    activities = loader.get_available_activities()
    
    print(f"\nFound {len(users)} users")
    if users:
        print(f"Sample users: {users[:5]}")
    
    print(f"Found {len(activities)} activities")
    if activities:
        print(f"Sample activities: {activities[:10]}")
    
    # load some sample data    
    if users and activities:
        user = users[0]
        activity = activities[0]
        samples = loader.load_user_data(user, activity, max_samples=5)
        
        print(f"\nSample data for user {user}, activity {activity}:")
        for i, sample in enumerate(samples):
            print(f"Sample {i+1}: {list(sample.keys())[:10]}")
            
            # print a few sensor values            
            sensor_values = {k: v for k, v in sample.items() 
                            if k not in ['user_id', 'activity', 'timestamp'] and i < 3}
            print(f"  Sensor values (sample): {list(sensor_values.items())[:5]}")
    
    # test synthetic data generation    
    print("\nGenerating synthetic data based on the dataset patterns...")
    synthetic_data = loader.generate_synthetic_data(5)
    print(f"Generated {len(synthetic_data)} synthetic samples")
    if synthetic_data:
        print(f"Sample synthetic data: {list(synthetic_data[0].keys())[:10]}")