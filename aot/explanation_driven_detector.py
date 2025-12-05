import json
from typing import Dict, List, Tuple, Any, Optional
import utilities as util
##2025DEC04##
#The index might be present in this code.  Verify, then remove before ingestion.  Also check syntax.  Output still says "user" and "performed".  This needs to be removed.

# for this, llm client can be anything. in our use-case, we went for ollama
class ExplanationDrivenDetector:
    def __init__(self, llm_client, coherence_threshold: float = 0.7):

        self.llm = llm_client
        self.coherence_threshold = coherence_threshold
    
    # this is where the sensor readings and metadata are needed to get textual representation of the pattern    
    def pattern_to_text(self, pattern: Dict[str, Any]) -> str:
        
        # extract metadata        
        #node_id = pattern.get('node_id', 'unknown')
        #subsystem = pattern.get('subsystem', 'unknown')
        #sensor = pattern.get('sensor', 'unknown')
        #parameter = pattern.get('parameter', 'unknown')
        #timestamp = pattern.get('timestamp', 'unknown')
        
        #text = f"Node {node_id} with {subsystem}, {sensor}, measuring {parameter} at {timestamp}\n\n"
        metadata,text,metadataKeys=util.extract_metadata(pattern)
        text += "Sensor readings:\n"
        
        # organize sensors by type        
        sensor_groups = {
            'value': [],
            'timestamp':[],
            'other':[]
        }
        #['node_id', 'subsystem', 'sensor', 'parameter']
        for key, value in pattern.items():
            if key not in metadataKeys['ids'] and key not in metadataKeys['labels'] and isinstance(value, (int, float)):
                if 'value' in key.lower():
                    sensor_groups['value'].append((key, value))
                elif 'timestamp' in key.lower():
                    sensor_groups['timestamp'].append((key, value))
                else:
                    sensor_groups['other'].append((key, value))
        
        # add organized sensor readings       
        for group, sensors in sensor_groups.items():
            if sensors:
                text += f"\n{group.upper()}:\n"
                for name, value in sensors:
                    text += f"- {name}: {value:.4f}\n"
        
        return text
    
    def context_to_text(self, retrieved_context: List[Dict[str, Any]]) -> str:
        """
        convert retrieved context to a textual representation.
        
        args:
            retrieved_context: list of dictionaries with retrieved similar patterns
            
        returns:
            text: textual representation of the retrieved context
        """
        if not retrieved_context:
            return "No similar patterns found in the database."
        
        text = f"Retrieved {len(retrieved_context)} similar patterns from the database:\n\n"
        
        # count normal and anomalous patterns        
        normal_patterns = [p for p in retrieved_context if not p.get('is_anomaly', False)]
        anomaly_patterns = [p for p in retrieved_context if p.get('is_anomaly', False)]
        
        text += f"Summary: {len(normal_patterns)} normal patterns, {len(anomaly_patterns)} anomaly patterns\n\n"
        
        # add information about closest normal pattern        
        if normal_patterns:
            closest_normal = min(normal_patterns, key=lambda x: x.get('distance', float('inf')))
            cn,cn_text,cn_keys=util.extract_metadata(closest_normal)
            text += "CLOSEST NORMAL PATTERN:\n"
            text += f"- Distance: {closest_normal.get('distance', 'unknown'):.4f}\n"
            for key in closest_normal.keys():
                if key in cn_keys['labels']:
                    text += f"- {key}: {closest_normal.get(key, 'unknown')}\n"       
            #text += f"- sybsystem: {closest_normal.get('subsystem', 'unknown')}\n"
            #text += f"- sensor: {closest_normal.get('sensor', 'unknown')}\n"
            #text += f"- parameter: {closest_normal.get('parameter', 'unknown')}\n"
            
            # add key sensor readings            
            for key, value in closest_normal.items():
                if 'value' in key and isinstance(value, (int, float)):
                    text += f"- {key}: {value:.4f}\n"
            
            # add key sensor timestamps            
            for key, value in closest_normal.items():
                if 'timestamp' in key:
                    text += f"- {key}: {value}\n"
            
            description = closest_normal.get('description', '')
            if description:
                text += f"- Description: {description}\n"
            
            text += "\n"
        
        # add information about closest anomaly pattern        
        if anomaly_patterns:
            closest_anomaly = min(anomaly_patterns, key=lambda x: x.get('distance', float('inf')))
            ca,ca_text,ca_keys=util.extract_metadata(closest_anomaly)
            text += "CLOSEST ANOMALY PATTERN:\n"
            text += f"- Distance: {closest_anomaly.get('distance', 'unknown'):.4f}\n"
            for key in closest_anomaly.keys():
                if key in ca_keys['labels']:
                    text += f"- {key}: {closest_anomaly.get(key, 'unknown')}\n"  
            #text += f"- sybsystem: {closest_anomaly.get('subsystem', 'unknown')}\n"
            #text += f"- sensor: {closest_anomaly.get('sensor', 'unknown')}\n"
            #text += f"- parameter: {closest_anomaly.get('parameter', 'unknown')}\n"
            text += f"- Anomaly Type: {closest_anomaly.get('anomaly_type', 'unknown')}\n"
            
            # add key sensor readings            
            for key, value in closest_anomaly.items():
                if key in ['value_hrf'] and isinstance(value, (int, float)):
                    text += f"- {key}: {value:.4f}\n"
            
            # add key sensor timestamps            
            for key, value in closest_anomaly.items():
                if 'timestamp' in key:
                    text += f"- {key}: {value}\n"
            
            explanation = closest_anomaly.get('explanation', '')
            if explanation:
                text += f"- Explanation: {explanation}\n"
        
        return text
    
    def construct_explain_prompt(self, pattern_text: str, context_text: str) -> str:
        """
        construct a prompt for explanation-driven detection.
        
        args:
            pattern_text: textual representation of the sensor pattern
            context_text: textual representation of the retrieved context
            
        returns:
            prompt: constructed prompt for the llm
        """
        prompt = f"""Perform explanation-driven anomaly detection on the following sensor data.

        CURRENT PATTERN:
        {pattern_text}

        CONTEXT FROM DATABASE:
        {context_text}

        TASK:
        Analyze the current pattern to determine if it represents an anomaly based on the context.
        Your explanation is crucial - the detection decision will be accepted only if your explanation is coherent and well-supported by the data.

        Provide your analysis as a JSON object with the following fields:
        1. "is_anomaly": true/false - whether this pattern represents an anomaly
        2. "confidence": a number between 0.0 and 1.0 - how confident you are in this assessment
        3. "explanation": a detailed explanation of your analysis, including specific sensor readings and why they are anomalous or normal
        4. "user_friendly_message": a simplified explanation for non-technical users
        5. "notable_deviations": a list of specific sensors with unusual readings (if any)
        6. "recommended_actions": suggested next steps based on your findings

        Format your entire response as a valid JSON object.
        """
        return prompt
    
    def evaluate_coherence(self, explanation: str, pattern: Dict[str, Any], retrieved_context: List[Dict[str, Any]]) -> float:
        """
        evaluate the coherence of the explanation with respect to the data.
        
        args:
            explanation: the explanation provided by the llm
            pattern: the current sensor pattern
            retrieved_context: retrieved similar patterns
            
        returns:
            coherence: a score between 0 and 1 indicating explanation coherence
        """
        # implement a simple coherence check based on:        
        # 1. whether the explanation references specific sensor values        
        # 2. whether those values actually appear in the pattern        
        # 3. whether the explanation makes comparisons to retrieved patterns        
        coherence_score = 0.0
        max_score = 3.0  # three criteria, each worth 1.0        
        # check if explanation references specific sensor values        
        sensor_refs = 0
        metadata,text,meta_keys=util.extract_metadata(pattern)
        #['node_id', 'subsystem','sensor','parameter']
        for key in pattern:
            if key not in meta_keys['ids'] and key not in meta_keys['labels'] and isinstance(pattern[key], (int, float)):
                if key in explanation.lower():
                    sensor_refs += 1
        
        # score based on sensor references (max 1.0)        
        if sensor_refs >= 3:
            coherence_score += 1.0
        else:
            coherence_score += sensor_refs / 3.0
        
        # check if numbers in the explanation match actual values        
        import re
        
        pattern_values = [v for k, v in pattern.items() 
                         if k not in meta_keys['ids'] and key not in meta_keys['labels'] 
                         and isinstance(v, (int, float))]
        
        # extract numbers from explanation        
        explanation_numbers = re.findall(r'\d+\.\d+', explanation)
        explanation_values = [float(num) for num in explanation_numbers]
        
        # count how many explanation values are close to actual values        
        value_matches = 0
        for ev in explanation_values:
            if any(abs(ev - pv) < 0.5 for pv in pattern_values):
                value_matches += 1
        
        # score based on value matches (max 1.0)        
        if explanation_values:
            coherence_score += min(1.0, value_matches / len(explanation_values))
        
        # check if explanation compares to retrieved patterns       
        comparison_terms = ['normal', 'typical', 'usual', 'average', 'expected', 
                           'similar', 'pattern', 'comparable', 'previous', 'historical']
        
        comparison_count = sum(1 for term in comparison_terms if term in explanation.lower())
        
        # score based on comparisons (max 1.0)        
        coherence_score += min(1.0, comparison_count / 3.0)
        
        # normalize to 0-1 range        
        return coherence_score / max_score
    
    def detect(self, 
              pattern: Dict[str, Any],
              retrieved_context: List[Dict[str, Any]]) -> Tuple[bool, float, str]:
        """
        perform explanation-driven anomaly detection as described in algorithm 4.
        
        args:
            pattern: sensor pattern to analyze
            retrieved_context: retrieved similar patterns
            
        returns:
            is_anomaly: whether the pattern is an anomaly
            confidence: confidence score (0 to 1)
            explanation: explanation of the analysis
        """
        # convert pattern to text        
        pattern_text = self.pattern_to_text(pattern)
        
        # convert context to text     
        context_text = self.context_to_text(retrieved_context)
        
        # construct prompt        
        prompt = self.construct_explain_prompt(pattern_text, context_text)
        
        # get llm analysis         
        llm_response = self.llm.generate(prompt)
        
        # parse llm response        
        try:
            result = json.loads(llm_response)
            is_anomaly = result.get('is_anomaly', False)
            confidence = float(result.get('confidence', 0.5))
            explanation = result.get('explanation', 'No explanation provided.')
            
            # include user-friendly message if available            
            user_friendly = result.get('user_friendly_message', '')
            if user_friendly:
                explanation = f"{explanation}\n\nUser-friendly explanation: {user_friendly}"
            
            # include notable deviations if available            
            notable_deviations = result.get('notable_deviations', [])
            if notable_deviations:
                if isinstance(notable_deviations, list):
                    explanation += "\n\nNotable deviations:\n- " + "\n- ".join(notable_deviations)
                else:
                    explanation += f"\n\nNotable deviations: {notable_deviations}"
            
            # include recommended actions if available            
            recommended_actions = result.get('recommended_actions', [])
            if recommended_actions:
                if isinstance(recommended_actions, list):
                    explanation += "\n\nRecommended actions:\n- " + "\n- ".join(recommended_actions)
                else:
                    explanation += f"\n\nRecommended actions: {recommended_actions}"
            
        except (json.JSONDecodeError, ValueError):
            # if parsing fails, extract information using simple heuristics            
            is_anomaly = 'anomaly' in llm_response.lower()
            confidence = 0.5
            explanation = llm_response[:500]  # truncate to a reasonable length        
        # evaluate coherence of the explanation   
        coherence = self.evaluate_coherence(explanation, pattern, retrieved_context)
        
        # make final decision based on coherence        
        if coherence < self.coherence_threshold:
            # reject detection if explanation incoherent            
            is_anomaly = False
            confidence = confidence * coherence
        
        return is_anomaly, confidence, explanation



if __name__ == "__main__":
    # I'm just using a mock llm client for testing    
    class MockLLM:
        def generate(self, prompt):
            # simulate llm response            
            return json.dumps({
                "is_anomaly": True,
                "confidence": 0.85,
                "explanation": "The current pattern shows significant deviations in accelerometer readings compared to normal walking patterns. The acc_y value of 8.5 is much lower than the expected 9.8 (gravity), and acc_x and acc_z values (0.5 and 0.9) are higher than usual (typically around 0.1-0.2).",
                "user_friendly_message": "Your walking pattern looks unusual. The way you're holding or moving with your phone is different from your normal walking.",
                "notable_deviations": [
                    "Accelerometer y-axis is 13% lower than normal",
                    "Accelerometer x-axis is 5x higher than normal",
                    "Accelerometer z-axis is 4.5x higher than normal"
                ],
                "recommended_actions": [
                    "Check if phone position has changed",
                    "Verify if walking surface is different",
                    "Monitor for consistent pattern changes"
                ]
            })
    
    # create sample data    
    pattern = {
        "user_id": "user123",
        "activity": "walking",
        "timestamp": "2023-01-01T12:00:00",
        "acc_x": 0.5,
        "acc_y": 8.5,
        "acc_z": 0.9,
        "gyro_x": 0.12,
        "gyro_y": 0.22,
        "gyro_z": 0.11
    }
    
    retrieved_context = [
        {
            "distance": 0.2,
            "is_anomaly": False,
            "activity": "walking",
            "description": "Normal walking pattern with regular gait",
            "acc_x": 0.1,
            "acc_y": 9.8,
            "acc_z": 0.2,
            "gyro_x": 0.01,
            "gyro_y": 0.02,
            "gyro_z": 0.01
        },
        {
            "distance": 0.8,
            "is_anomaly": True,
            "activity": "walking",
            "anomaly_type": "behavioral",
            "explanation": "Irregular walking pattern showing unusual acceleration",
            "acc_x": 0.4,
            "acc_y": 8.7,
            "acc_z": 0.7,
            "gyro_x": 0.10,
            "gyro_y": 0.20,
            "gyro_z": 0.10
        }
    ]
    
    # initialize detector with mock llm    
    detector = ExplanationDrivenDetector(MockLLM())
    
    # perform detection    
    is_anomaly, confidence, explanation = detector.detect(pattern, retrieved_context)
      
    print("Explanation-Driven Detection Result:")
    print(f"Is Anomaly: {is_anomaly}")
    print(f"Confidence: {confidence:.4f}")
    print(f"Explanation:\n{explanation}")