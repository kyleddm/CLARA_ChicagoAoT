import json
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import utilities as util
from utilities import logger
class ContextualDeviationAnalyzer:
    
    def __init__(self, llm_client, args):
        self.llm = llm_client
        self.args=args
        self.log_file=self.args.output_dir+"log.txt"
    def pattern_to_text(self, pattern: Dict[str, Any]) -> str:
        metadata,text,meta_keys=util.extract_metadata(pattern, self.args)
        # extract metadata        
        #host_id = pattern.get('hostID', 'unknown')
        #activity = pattern.get('activity', 'unknown')
        #timestamp = pattern.get('timestamp', 'unknown')
        
        #text = f"Host {host_id} performing {activity} at {timestamp}\n"
        text += "Sensor readings:\n"
        
        # add sensor readings        
        for key, value in pattern.items():
            #['hostID', 'activity', 'timestamp', 'uuid']
            if key in meta_keys['values'] and isinstance(value, (int, float)):
                text += f"- {key}: {value:.4f}\n"
            else:
                text += f"- {key}: {value}\n"
        
        return text
    
    def context_to_text(self, context: Dict[str, Any]) -> str: #this looks to be a redundant function now with the utility component added.
        text = f"Context:\n"
        metadata,text2,meta_keys=util.extract_metadata(context, self.args)
        #node_id = context.get('node_id', 'unknown')
        #activity = context.get('activity', 'unknown')
        text += text2
        #for key in meta_keys['ids']:
        #    text += f"- Host with id {key}: {context.get(key, 'unknown')}\n"
        #text += f"sensor-related identification: \n"
        #for key in meta_keys['labels']:
        #    text += f"- {key}: {context.get(key, 'unknown')}\n"
         #"Host {host_id} performing {activity}\n"
        #text += f"sensor reradings: \n"
        # add additional context information        
        for key, value in context.items():
            #not in ['hostID', 'activity']
            if key in meta_keys['values'] and isinstance(value, (str, int, float, bool)):
                text += f"- {key}: {value}\n"
        
        return text
    
    def retrieved_to_text(self, retrieved_patterns: List[Dict[str, Any]]) -> str:
        
        if not retrieved_patterns:
            return "No similar patterns found in the database."
        
        text = f"Retrieved {len(retrieved_patterns)} similar patterns:\n\n"
        
        # sort by similarity (distance)        
        sorted_patterns = sorted(retrieved_patterns, key=lambda x: x.get('distance', float('inf')))
        
        for i, pattern in enumerate(sorted_patterns): 
            #metadata,text2,meta_keys=util.extract_metadata(pattern, self.args)            #Don't think this is needed here.  I think it's double-extracting
            distance = pattern.get('distance', 'unknown')
            text += f"Pattern {i+1} (Distance: {distance:.4f}):\n"
            is_anomaly = pattern.get('is_anomaly', False)
            #activity = pattern.get('activity', 'unknown')
            for key in pattern['labels']:
                if key.lower() != 'distance' and key.lower() != 'is_anomaly':
                    text += f"- {key}: {pattern.get(key, 'unknown '+str(key))}\n"
            
            #text += f"- Activity: {activity}\n"
            text += f"- Is Anomaly: {'Yes' if is_anomaly else 'No'}\n"
            
            # add sensor readings            
            for key, value in pattern.items():
                #['user_id', 'activity', 'timestamp', 'distance', 'is_anomaly', 'description', 'explanation']
                if key not in pattern['ids'] and key not in pattern['labels'] and isinstance(value, (int, float)):
                    text += f"- {key}: {value:.4f}\n"
            
            description = pattern.get('description', '') or pattern.get('explanation', '')
            if description:
                text += f"- Description: {description}\n"
            
            text += "\n"
        
        return text
    
    def construct_prompt(self, pattern_text: str, context_text: str, retrieved_text: str) -> str:
        redundant_info=f"""        
        CONTEXT INFORMATION:
        {context_text}"""

        prompt = f"""Perform a contextual deviation analysis on the following sensor data.

        CURRENT PATTERN WITH CONTEXT:
        {pattern_text}

        SIMILAR PATTERNS FROM DATABASE:
        {retrieved_text}

        TASK:
        Analyze whether the current pattern deviates significantly from what would be expected given the context and similar patterns.

        Provide your analysis as a JSON object with the following fields:
        1. "is_anomaly": true/false - whether this pattern represents an anomaly
        2. "confidence": a number between 0.0 and 1.0 - how confident you are in this assessment
        3. "explanation": a detailed explanation of your analysis, including specific deviations

        Format your entire response as a valid JSON object.

        Remember, you are explicitly analyzing the data represented by the input pattern: {pattern_text}, using the similar reference patterns provided.  Make sure the analysis response is in reference to the input pattern and is not performed on one of the reference patterns.  If the reference patterns do not match the context provided with the input pattern, disregard them in yor analysis, but in your explaination, state which patterns were disregarded by including their ID and distance calculations.  When comparing the input pattern to the reference patterns, do not treat ID, distance, timestamp, or value keys as context labels.  If the only available reference patterns have different IDs, keep that in mind; different IDs may indicate the data is in a different location.  If the distances recorded are extremly large, disregard these patterns as well.
        """
        return prompt
    
    def normalize_score(self, score: float) -> float:

        # ensure score is between 0 and 1       
        return max(0.0, min(1.0, score))
    
    def analyze(self, 
               pattern: Dict[str, Any], 
               context: Dict[str, Any],
               retrieved_patterns: List[Dict[str, Any]]) -> Tuple[float, str]:

        # convert pattern to text         
        pattern_text = self.pattern_to_text(pattern)
        logger(self.log_file,f"CDA-Pattern Text:{pattern_text}")
        # convert context to text         
        context_text = self.context_to_text(context)
        logger(self.log_file,f"CDA-Context Text:{context_text}")
        # convert retrieved patterns to text         
        retrieved_text = self.retrieved_to_text(retrieved_patterns)
        logger(self.log_file,f"CDA-Retreived Text:{retrieved_text}")
        # construct prompt         
        prompt = self.construct_prompt(pattern_text, context_text, retrieved_text)
        logger(self.log_file,f"CDA-Prompt{prompt}")
        # get llm analysis         
        llm_response = self.llm.generate(prompt)
        logger(self.log_file,f"CDA-LLM Response:{llm_response}")
        # parse llm response        
        try:
            result = json.loads(llm_response)
            #check the syntax here, this is an arbitrary value that seems to get set to 0 if the output is nonamonalous
            anomaly_score = float(result.get('confidence', -999)) #if result.get('is_anomaly', False) else 0.0
            explanation = result.get('explanation', 'No explanation provided by Contextual Deviation Analyzer.')
        except (json.JSONDecodeError, ValueError):
            # if parsing fails, extract information using simple heuristics arbitrarily set confidence, but print whole response            
            anomaly_score = -999 #if 'anomaly' in llm_response.lower() else 0.0
            explanation = llm_response
            logger(self.log_file,f"LLM RESPONSE JSON DECODE ERROR, SCORE ARBITRARILY SET")            
        normalized_score = self.normalize_score(anomaly_score)
        
        logger(self.log_file,f"CDA-Normalized Score:{normalized_score}")
        logger(self.log_file,f"CDA-Reported Explanation:{explanation}")
        return normalized_score, explanation