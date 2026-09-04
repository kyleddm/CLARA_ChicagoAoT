import json
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
import utilities as util
from utilities import logger
class SensorDataAugmenter:
    
    def __init__(self, args, max_token_limit: int = 2048):

        self.max_token_limit = max_token_limit
        self.args=args
        self.log_file=self.args.output_dir+"log.txt"
    
    def sensor_to_text(self, sensor_data: Dict[str, Any]) -> str:

        # extract metadata
        metadata,text2,meta_keys=util.extract_metadata(sensor_data, self.args)
        text=text2
        #host_id = sensor_data.get('hostID', 'unknown')
        #activity = sensor_data.get('activity', 'unknown')
        #timestamp = sensor_data.get('timestamp', '')
        
        # start with a high-level description        
        #text = f"Host {host_id} performing {activity} at {timestamp}.\n\n"
        text += "Sensor readings:\n"
        
        # group sensors by type        
        sensor_groups = self._group_sensors(sensor_data)
        #print (f'TEST~~~SENSOR GROUPS: {sensor_groups}~~~TEST~~~\n')#THIS PRINTS NOTHING!  TEST THIS!!!
        # add each sensor group's readings        
        for group_name, sensors in sensor_groups.items():
            if sensors:
                text += f'\n{group_name.upper()} SENSORS:\n'
                for sensor_name, value in sensors:
                    text += f'- {sensor_name}: {value}\n'
        logger(self.log_file,f"SDA-Sensor To Text: {text}")
        return text
    
    def _group_sensors(self, sensor_data: Dict[str, Any]) -> Dict[str, List[Tuple[str, float]]]:
        metadata,text2,meta_keys=util.extract_metadata(sensor_data,self.args)
        sensor_groups=util.extractSensorType(sensor_data)
        
        # remove empty groups        
        return {k: v for k, v in sensor_groups.items() if v}
    
    def context_to_text(self, retrieved_context: List[Dict[str, Any]]) -> str:

        if not retrieved_context:
            return "No similar patterns found in the database."
        
        text = "Similar patterns found in the database:\n\n"
        
        # sort by similarity (distance)        
        sorted_context = sorted(retrieved_context, key=lambda x: x.get('distance', float('inf')))
        
        for i, pattern in enumerate(sorted_context[:5]):  # limit to top 5 for clarity
            #print(f'CONTEXT TO TEXT CHECK!! pattern: {pattern}')
            #metadata,text2,meta_keys=util.extract_metadata(pattern,self.args)#this might not be needed here; I may be re-extracting data already stored.
            distance = pattern.get('distance', 'unknown')
            is_anomaly = pattern.get('is_anomaly', False)
            #activity = pattern.get('activity', 'unknown')
            description = pattern.get('description', '') or pattern.get('explanation', '')
            #for id in pattern['ids']:
                #text += f"ID: {id}\n"
            #for key in pattern['labels']:
                #if key.lower() != 'distance' and key.lower() != 'is_anomaly' and key.lower() != 'description' and key.lower() != 'explanation':
                    #text += f"- {key}: {pattern.get(key, 'unknown '+str(key))}\n"
                        # add sensor readings            
            for key, value in pattern.items():
                if key in pattern['ids']:
                    text += f"ID: {id}\n"
                if key in pattern['labels']:
                    if key.lower() != 'distance' and key.lower() != 'is_anomaly' and key.lower() != 'description' and key.lower() != 'explanation':
                        text += f"- context label: {key}\n"
                #['user_id', 'activity', 'timestamp', 'distance', 'is_anomaly', 'description', 'explanation']
                if key not in pattern['ids'] and key not in pattern['labels'] and isinstance(value, (int, float)):
                    text += f"- {key}: {value:.4f}\n"
            text += f"PATTERN {i+1} (Distance: {distance:.4f}):\n"
            #text += f"- Activity: {activity}\n"
            text += f"- Is Anomaly: {'Yes' if is_anomaly else 'No'}\n"
            if description:
                text += f"- Description: {description}\n"
            text += "\n"
        logger(self.log_file,f"SDA-Context to text:{text}")
        return text
    
    def generate_guidance(self, sensor_data: Dict[str, Any], 
                         retrieved_context: List[Dict[str, Any]]) -> str:#needs to be edited.  Activity and user_id are not generalized
        metadata,text2,meta_keys=util.extract_metadata(sensor_data, self.args)
        activity=[]        
        for key, value in sorted(sensor_data.items()):
            if key in meta_keys['ids']:
                user_id=sensor_data[key]
            if key in meta_keys['labels']:
                activity.append(sensor_data[key])
        # determine user activity and context       
        #activity = sensor_data.get('activity', 'unknown activity')
        #user_id = sensor_data.get('user_id', 'unknown user')
        
        # find anomalies in retrieved context        
        anomalies = [c for c in retrieved_context if c.get('is_anomaly', False)]
        
        guidance = "ANALYSIS GUIDANCE:\n\n"
        
        # add activity-specific guidance        
        guidance += f"When analyzing this data, consider that the node {user_id} is definied by the following labels: '{activity}'.\n"
        
        # add context-based guidance        
        if anomalies:
            guidance += f"There are {len(anomalies)} similar anomalous patterns in the database.\n"
            guidance += "Pay special attention to these patterns when making your assessment.\n"
        else:
            guidance += "No similar anomalies found in the database.\n"
            guidance += "Focus on detecting deviations from the normal patterns.\n"
        
        guidance+=util.provideGuidance(metadata)
        logger(self.log_file,f"SDA-Guidance:{guidance}")
        return guidance
    
    def highlight_patterns(self, sensor_data: Dict[str, Any], 
                          retrieved_context: List[Dict[str, Any]]) -> str:

        highlights = "KEY PATTERNS AND POTENTIAL ANOMALIES:\n\n"
        
        # find the most similar normal pattern        
        normal_patterns = [c for c in retrieved_context if not c.get('is_anomaly', False)]
        
        if normal_patterns:
            closest_normal = min(normal_patterns, key=lambda x: x.get('distance', float('inf')))
            
            # compare sensor readings with closest normal pattern           
            highlights += "Comparison with closest normal pattern:\n"
            
            for key, value in sensor_data.items():
                if key not in ['user_id', 'activity', 'timestamp', 'uuid'] and isinstance(value, (int, float)):
                    normal_value = closest_normal.get(key)
                    if normal_value is not None and isinstance(normal_value, (int, float)):
                        diff_pct = ((value - normal_value) / normal_value * 100) if normal_value != 0 else float('inf')
                        
                        if abs(diff_pct) > 50:
                            highlights += f"- {key}: Current {value:.4f} vs Normal {normal_value:.4f} "
                            highlights += f"({diff_pct:+.1f}%) [SIGNIFICANT DEVIATION]\n"
                        elif abs(diff_pct) > 20:
                            highlights += f"- {key}: Current {value:.4f} vs Normal {normal_value:.4f} "
                            highlights += f"({diff_pct:+.1f}%) [MODERATE DEVIATION]\n"
        else:
            highlights += "No normal patterns found for comparison.\n"
        
        # check for known anomaly patterns        
        anomaly_patterns = [c for c in retrieved_context if c.get('is_anomaly', False)]
        
        if anomaly_patterns:
            closest_anomaly = min(anomaly_patterns, key=lambda x: x.get('distance', float('inf')))
            
            highlights += "\nSimilarity to known anomaly:\n"
            highlights += f"- Distance: {closest_anomaly.get('distance', 'unknown'):.4f}\n"
            highlights += f"- Type: {closest_anomaly.get('anomaly_type', 'unknown')}\n"
            explanation = closest_anomaly.get('explanation', '') or closest_anomaly.get('description', '')
            if explanation:
                highlights += f"- Description: {explanation}\n"
        logger(self.log_file,f"SDA- Highlights:{highlights}")
        return highlights
    
    def augment_sensor_data(self, sensor_data: Dict[str, Any], 
                           retrieved_context: List[Dict[str, Any]]) -> str:
        # convert sensor data to text     
        sensor_text = self.sensor_to_text(sensor_data)
        
        # structure retrieved context     
        context_text = self.context_to_text(retrieved_context)
        
        # generate guidance for llm        
        guidance_text = self.generate_guidance(sensor_data, retrieved_context)
        
        # highlight key patterns     
        pattern_text = self.highlight_patterns(sensor_data, retrieved_context)
        
        # combine all components       
        prompt = f"""SENSOR DATA:
                {sensor_text}

                RETRIEVED CONTEXT:
                {context_text}

                {guidance_text}

                {pattern_text}

                INSTRUCTIONS:
                Based on the above information, please analyze if the current sensor data represents an anomaly.
                Provide your analysis in JSON format with the following fields:
                - "is_anomaly": true/false
                - "confidence": a value between 0.0 and 1.0
                - "anomaly_type": "behavioral", "technical", or "unknown" (if is_anomaly is true)
                - "explanation": A detailed explanation of your analysis
                - "user_friendly_message": A simple explanation that a regular user could understand
                - "notable_deviations": List of specific sensors with unusual readings
                - "recommended_actions": Suggestions for addressing the anomaly
                """
        logger(self.log_file,f"SDA-Augmentation of prompt:{prompt}")
        return prompt