import json
import requests
from typing import Dict, List, Any, Optional

class OllamaLLM:
    # client for interacting with llama-3.2-1b model via ollama api
    def __init__(self, 
                 model_name: str = "llama3.2:1b", 
                 api_base: str = "http://localhost:11434",
                 temperature: float = 0.7,
                 max_tokens: int = 1024):

        self.model_name = model_name
        self.api_base = api_base
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.api_url = f"{api_base}/api/generate"
    
    def generate(self, prompt: str) -> str:

        try:
            payload = {
                "model": self.model_name,
                "prompt": prompt,
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
                "stream": False
            }
            
            response = requests.post(self.api_url, json=payload)
            response.raise_for_status()
            result = response.json()
            
            return result.get("response", "")
        except requests.exceptions.RequestException as e:
            print(f"Error communicating with Ollama API: {e}")
            return f"Error: {str(e)}"
    
    def analyze_anomaly(self, 
                       current_data: Dict[str, Any], 
                       similar_patterns: List[Dict[str, Any]],
                       threshold: float) -> Dict[str, Any]:

        # construct the prompt for anomaly analysis        
        prompt = self._construct_anomaly_prompt(current_data, similar_patterns, threshold)
        
        # get llm response        
        llm_response = self.generate(prompt)
        
        # parse the llm response        
        return self._parse_analysis_response(llm_response, similar_patterns)
    
    def _construct_anomaly_prompt(self, 
                                 current_data: Dict[str, Any], 
                                 similar_patterns: List[Dict[str, Any]],
                                 threshold: float) -> str:

        # format current sensor data        
        current_data_str = json.dumps(current_data, indent=2)
        
        # format similar patterns        
        similar_patterns_str = json.dumps(similar_patterns, indent=2)
        
        # extract key sensors for focused analysis        
        key_sensors = []
        for key in current_data:
            if any(sensor_type in key.lower() for sensor_type in ['acc', 'gyro', 'mag', 'rotation']):
                if isinstance(current_data[key], (int, float)):
                    key_sensors.append(key)
        
        key_sensors_str = ", ".join(key_sensors[:10])  # I set this to for demo 10 to avoid overlong prompt        
        # determine activity context        
        activity = current_data.get('activity', 'unknown activity')
        hostID = current_data.get('hostID', 'unknown user')
        
        # construct the prompt        
        prompt = f"""You are an expert system analyzing mobile device sensor data for anomalies. You need to provide clear, human-understandable explanations about your findings.

                    ## current sensor data:```json
                    {current_data_str}
                    ```

                    ## retrieved similar patterns:```json
                    {similar_patterns_str}
                    ```

                    ## context:- The current data represents sensors from user "{hostID}" performing "{activity}"
                    - Key sensors to focus on: {key_sensors_str}
                    - The similarity threshold is {threshold} (distances larger than this suggest anomalies)

                    ## instructions:
                    1. Analyze the current sensor data and compare it with the retrieved similar patterns.

                    2. Determine if the current data represents an anomaly based on:
                    - Similarity distance to known patterns (anomaly if distance > {threshold})
                    - Comparison with known anomalous patterns
                    - Significant deviations in sensor readings

                    3. Provide the following in your response in JSON format:
                    - "is_anomaly": true/false
                    - "confidence": a value between 0.0 and 1.0 (higher = more confident)
                    - "anomaly_type": "behavioral", "technical", or "unknown" (if is_anomaly is true)
                    - "explanation": A detailed, human-readable explanation of your analysis
                    - "user_friendly_message": A simple, non-technical explanation that a regular user could understand
                    - "notable_deviations": List the specific sensors with unusual readings and how they differ from normal patterns
                    - "recommended_actions": Suggestions for what the user or system should do next

                    Make sure the "explanation" and "user_friendly_message" are clear, concise, and helpful for understanding the anomaly detection result. The explanation should be technical but understandable, while the user_friendly_message should avoid technical jargon completely.

                    Format your entire response as a valid JSON object.
                    """
        return prompt
    
    def _parse_analysis_response(self, 
                                llm_response: str, 
                                similar_patterns: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        parse the llm response into a structured analysis result.
        
        args:
            llm_response: raw response from the llm
            similar_patterns: retrieved similar patterns from vector database
            
        returns:
            result: structured analysis result
        """
        try:
            # try to parse the response as json            
            result = json.loads(llm_response)
            
            # ensure all required fields are present            
            if not all(k in result for k in ["is_anomaly", "confidence", "explanation"]):
                # if missing fields, create a default structure                
                result = {
                    "is_anomaly": False,
                    "confidence": 0.0,
                    "anomaly_type": None,
                    "explanation": "Failed to parse LLM response properly. Raw response: " + llm_response[:100] + "..."
                }
            
            # add similar patterns to the result           
            result["similar_patterns"] = similar_patterns
            
            return result
        except json.JSONDecodeError:
            # if json parsing fails, extract information using regex or other parsing methods 
            # for demo purposes, return a default result with the raw response           
            return {
                "is_anomaly": "anomaly" in llm_response.lower(),
                "confidence": 0.5,
                "anomaly_type": "unknown",
                "explanation": "Failed to parse LLM response as JSON. Raw response: " + llm_response[:200] + "...",
                "similar_patterns": similar_patterns
            }