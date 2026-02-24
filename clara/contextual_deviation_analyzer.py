import json
from typing import Dict, List, Tuple, Any, Optional
import numpy as np

class ContextualDeviationAnalyzer:
    
    def __init__(self, llm_client):
        self.llm = llm_client
    
    def pattern_to_text(self, pattern: Dict[str, Any]) -> str:
  
        # extract metadata        
        host_id = pattern.get('hostID', 'unknown')
        activity = pattern.get('activity', 'unknown')
        timestamp = pattern.get('timestamp', 'unknown')
        
        text = f"Host {host_id} performing {activity} at {timestamp}\n"
        text += "Sensor readings:\n"
        
        # add sensor readings        
        for key, value in pattern.items():
            if key not in ['hostID', 'activity', 'timestamp', 'uuid'] and isinstance(value, (int, float)):
                text += f"- {key}: {value:.4f}\n"
        
        return text
    
    def context_to_text(self, context: Dict[str, Any]) -> str:

        host_id = context.get('hostID', 'unknown')
        activity = context.get('activity', 'unknown')
        
        text = f"Context: Host {host_id} performing {activity}\n"
        
        # add additional context information        
        for key, value in context.items():
            if key not in ['hostID', 'activity'] and isinstance(value, (str, int, float, bool)):
                text += f"- {key}: {value}\n"
        
        return text
    
    def retrieved_to_text(self, retrieved_patterns: List[Dict[str, Any]]) -> str:
        
        if not retrieved_patterns:
            return "No similar patterns found in the database."
        
        text = f"Retrieved {len(retrieved_patterns)} similar patterns:\n\n"
        
        # sort by similarity (distance)        
        sorted_patterns = sorted(retrieved_patterns, key=lambda x: x.get('distance', float('inf')))
        
        for i, pattern in enumerate(sorted_patterns[:3]):  # limit to top 3 for clarity            
            distance = pattern.get('distance', 'unknown')
            is_anomaly = pattern.get('is_anomaly', False)
            activity = pattern.get('activity', 'unknown')
            
            text += f"Pattern {i+1} (Distance: {distance:.4f}):\n"
            text += f"- Activity: {activity}\n"
            text += f"- Is Anomaly: {'Yes' if is_anomaly else 'No'}\n"
            
            # add sensor readings            
            for key, value in pattern.items():
                if key not in ['user_id', 'activity', 'timestamp', 'distance', 'is_anomaly', 'description', 'explanation'] and isinstance(value, (int, float)):
                    text += f"- {key}: {value:.4f}\n"
            
            description = pattern.get('description', '') or pattern.get('explanation', '')
            if description:
                text += f"- Description: {description}\n"
            
            text += "\n"
        
        return text
    
    def construct_prompt(self, pattern_text: str, context_text: str, retrieved_text: str) -> str:

        prompt = f"""Perform a contextual deviation analysis on the following sensor data.

        CURRENT PATTERN:
        {pattern_text}

        CONTEXT INFORMATION:
        {context_text}

        SIMILAR PATTERNS FROM DATABASE:
        {retrieved_text}

        TASK:
        Analyze whether the current pattern deviates significantly from what would be expected given the context and similar patterns.

        Provide your analysis as a JSON object with the following fields:
        1. "is_anomaly": true/false - whether this pattern represents an anomaly
        2. "confidence": a number between 0.0 and 1.0 - how confident you are in this assessment
        3. "explanation": a detailed explanation of your analysis, including specific deviations

        Format your entire response as a valid JSON object.
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
        
        # convert context to text         
        context_text = self.context_to_text(context)
        
        # convert retrieved patterns to text         
        retrieved_text = self.retrieved_to_text(retrieved_patterns)
        
        # construct prompt         
        prompt = self.construct_prompt(pattern_text, context_text, retrieved_text)
        
        # get llm analysis         
        llm_response = self.llm.generate(prompt)
        
        # parse llm response        
        try:
            result = json.loads(llm_response)
            anomaly_score = float(result.get('confidence', 0.5)) if result.get('is_anomaly', False) else 0.0
            explanation = result.get('explanation', 'No explanation provided.')
        except (json.JSONDecodeError, ValueError):
            # if parsing fails, extract information using simple heuristics            
            anomaly_score = 0.5 if 'anomaly' in llm_response.lower() else 0.0
            explanation = llm_response[:500]  # truncate to a reasonable length              
        normalized_score = self.normalize_score(anomaly_score)
        
        return normalized_score, explanation


if __name__ == "__main__":
    # mock llm client for testing    
    class MockLLM:
        def generate(self, prompt):
            # simulate llm response            
            return json.dumps({
                "is_anomaly": True,
                "confidence": 0.85,
                "explanation": "The current pattern shows significant deviations in accelerometer readings compared to normal walking patterns. The y-axis value is much lower than expected, and x/z values are higher than usual."
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
    
    context = {
        "user_id": "user123",
        "activity": "walking",
        "time_of_day": "morning",
        "location": "home"
    }
    
    retrieved_patterns = [
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
            "explanation": "Irregular walking pattern showing unusual acceleration",
            "acc_x": 0.4,
            "acc_y": 8.7,
            "acc_z": 0.7,
            "gyro_x": 0.10,
            "gyro_y": 0.20,
            "gyro_z": 0.10
        }
    ]
    
    # initialize analyzer with mock llm (I'm just using a mock llm client for testing)    
    analyzer = ContextualDeviationAnalyzer(MockLLM())
    
    
    anomaly_score, explanation = analyzer.analyze(pattern, context, retrieved_patterns)
    
       
    print(f"Contextual Deviation Analysis Result:")
    print(f"Anomaly Score: {anomaly_score:.4f}")
    print(f"Explanation: {explanation}")