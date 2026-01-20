import numpy as np
import json
import os
import time
from typing import Dict, List, Tuple, Any, Optional
import torch
import utilities as util

from clara.faiss_vector_store import FAISSVectorStore
from clara.ollama_llm import OllamaLLM
from .contrastive_embedding_model import generate_sensor_embeddings_with_contrastive_learning
from .sensor_data_augmenter import SensorDataAugmenter
from .contextual_deviation_analyzer import ContextualDeviationAnalyzer
from .explanation_driven_detector import ExplanationDrivenDetector

class CLARA:
    def __init__(self, arguments, vector_store_path: str = None, llm_model_name: str = "llama3.2:1b", llm_api_base: str = "http://localhost:11434", embedding_dim: int = 768, semantic_threshold: float = 0.7, coherence_threshold: float = 0.7):
            # vector_store_path: path to a saved faiss index file (optional)            
            # llm_model_name: name of the llama model in ollama            
            # llm_api_base: base url for ollama api           
            # embedding_dim: dimensionality of the embedding vectors            
            # semantic_threshold: threshold for semantic pattern matching            
            # coherence_threshold: threshold for explanation coherence            
        self.llm = OllamaLLM(
            model_name=llm_model_name,
            api_base=llm_api_base
        )
        self.args=arguments
        self.vector_store = FAISSVectorStore(
            dimension=embedding_dim,
            index_type="flat"  # i'm using flat for exact search. todo: explore more options in the future.         
            )
        
        if vector_store_path and os.path.exists(vector_store_path):
            self.vector_store.load(vector_store_path)

        self.augmenter = SensorDataAugmenter(self.args)
        self.deviation_analyzer = ContextualDeviationAnalyzer(self.llm,self.args)
        self.explanation_detector = ExplanationDrivenDetector(self.llm, self.args, coherence_threshold)#2026JAN14; added args.  there might be an error in the code here.  coherence_threshold isn't a mandatory argument, but args is for ExplainationDrivenDetector.  this means the code likely set args as the coherence threshold, and used the default CT for all other computation.
        
    
        self.embedding_dim = embedding_dim
        self.semantic_threshold = semantic_threshold
        self.coherence_threshold = coherence_threshold
        
        self.embedding_model = None
    
    def _sensor_to_features(self, sensor_data: Dict[str, Any]) -> np.ndarray:

        features = []
        #print(f'sensor data: {sensor_data}')
        #  sensor readings in dictionary data format. tried multi-dimensional list, but it was taking too long.
        metadata,text2,meta_keys=util.extract_metadata(sensor_data, self.args)         
        for key, value in sorted(sensor_data.items()):
            if key not in meta_keys['ids'] and key not in meta_keys['labels'] and isinstance(value, (int, float)):
            #if key not in ['user_id', 'activity', 'timestamp', 'uuid'] and isinstance(value, (int, float)):
                features.append(float(value))
        if not features:
            features = [0.0] * 10
        return np.array(features, dtype=np.float32)
    
    
    def _create_embedding(self, sensor_data: Dict[str, Any]) -> np.ndarray:
        
        features = self._sensor_to_features(sensor_data)

        #print(f"Feature length: {len(features)}")
        # if we have a trained embedding model, use it       
        if self.embedding_model:
            try:
                with torch.no_grad():
                    features_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0)
                    embedding = self.embedding_model(features_tensor).cpu().numpy()
                
                # check embedding shape                
                print(f"Embedding from model shape: {embedding.shape}")
                
                
                if embedding.shape[1] != self.embedding_dim:
                    print(f"WARNING: Embedding dimension mismatch. Expected {self.embedding_dim}, got {embedding.shape[1]}. Adjusting...")
                    if embedding.shape[1] > self.embedding_dim:
                        embedding = embedding[:, :self.embedding_dim]
                    else:
                        padded = np.zeros((embedding.shape[0], self.embedding_dim), dtype=np.float32)
                        padded[:, :embedding.shape[1]] = embedding
                        embedding = padded
                
                return embedding.astype(np.float32)
            except Exception as e:
                print(f"Error using embedding model: {e}")
                print("Falling back to simple embedding")
        
        embedding = np.zeros((1, self.embedding_dim), dtype=np.float32)
        
        feature_len = len(features)
        if feature_len > 0:
            repeat_times = self.embedding_dim // feature_len + 1
            repeated_features = np.tile(features, repeat_times)
            embedding[0, :self.embedding_dim] = repeated_features[:self.embedding_dim]

        #print(f"Simple embedding shape: {embedding.shape}")
        
        return embedding
    
    def train_embedding_model(self, sensor_data_list: List[Dict[str, Any]]) -> None:
        # extract features and labels        
        features = []
        labels = []
        
        for data in sensor_data_list:
            feature_vector = self._sensor_to_features(data)
            features.append(feature_vector)
            metadata,text2,meta_keys=util.extract_metadata(data, self.args)
            #activity = data.get('activity', 'unknown')
            for label in metadata['labels']:
                labels.append(hash(label) % 100)  
        features_array = np.array(features, dtype=np.float32)
        labels_array = np.array(labels, dtype=np.int64)
        
        print(f"Training contrastive embedding model on {len(features)} samples...")
        
        # use contrastive learning to train the embedding model        
        self.embedding_model, _ = generate_sensor_embeddings_with_contrastive_learning(
            feature_vectors=features_array,
            labels=labels_array,
            embedding_dim=self.embedding_dim,
            epochs=100
        )
        
        print("Embedding model training complete!")
    
    
    # important: takes sensor_data and description. it gives the id of the added pattern in the vector store.    
    def add_normal_pattern(self, sensor_data: Dict[str, Any], description: str = None) -> int:
        
        # create embedding        
        embedding = self._create_embedding(sensor_data)
        metadata={}
        metadata,text2,meta_keys=util.extract_metadata(sensor_data, self.args)
        metadata['is_anomaly']=False
        metadata['description']=description or 'Normal sensor pattern'
        # prepare metadata        
        #metadata = {
        #    "hostID": sensor_data.get("hostID", "unknown"),
        #    "activity": sensor_data.get("activity", "unknown"),
        #    "timestamp": sensor_data.get("timestamp", ""),
        #    "is_anomaly": False,
        #    "description": description or "Normal sensor pattern"
        #}
        
        # add sensor readings to metadata for comparison        
        for key, value in sensor_data.items():
            #if key not in ['hostID', 'activity', 'timestamp', 'uuid'] and isinstance(value, (int, float)):
            if key in meta_keys['values'] and isinstance(value, (int, float)):
                metadata[key] = value
            if key in meta_keys['timestamps']:
                timVal=str(value)
                #timVal=util.pruneTime(str(value))
                metadata[key] = timVal
        
        # add to vector store       
        pattern_id = self.vector_store.add_embedding(embedding, metadata)
        
        return pattern_id
    
    def add_anomaly_pattern(self, 
                           sensor_data: Dict[str, Any], 
                           anomaly_type: str, 
                           explanation: str) -> int:
        """
        add an anomalous sensor pattern to the vector store.
        
        args:
            sensor_data: dictionary containing sensor readings and metadata
            anomaly_type: type of anomaly (e.g., "behavioral", "technical")
            explanation: explanation of why this pattern is anomalous
            
        returns:
            id: id of the added pattern in the vector store
        """
        
        embedding = self._create_embedding(sensor_data)
        metadata,text2,meta_keys=util.extract_metadata(sensor_data, self.args)
        metadata['is_anomaly']=True
        metadata['anomaly_type']=anomaly_type
        metadata["explanation"]=explanation
        #metadata = {
        #    "hostID": sensor_data.get("hostID", "unknown"),
        #    "activity": sensor_data.get("activity", "unknown"),
        #    "timestamp": sensor_data.get("timestamp", ""),
        #    "is_anomaly": True,
        #    "anomaly_type": anomaly_type,
        #    "explanation": explanation
        #}
        
        # include sensor readings to metadata for comparison        
        #for key, value in sensor_data.items():
        #    if key not in ['hostID', 'activity', 'timestamp', 'uuid'] and isinstance(value, (int, float)):
        #        metadata[key] = value
                # add sensor readings to metadata for comparison        
        for key, value in sensor_data.items():
            #if key not in ['hostID', 'activity', 'timestamp', 'uuid'] and isinstance(value, (int, float)):
            if key in meta_keys['values'] and isinstance(value, (int, float)):
                metadata[key] = value
            if key in meta_keys['timestamps']:
                timVal=str(value)
                #timVal=util.pruneTime(str(value))
                metadata[key] = timVal
        
        # send it to db        
        pattern_id = self.vector_store.add_embedding(embedding, metadata)
        
        return pattern_id
    
    def multi_query_retrieval(self, sensor_data: Dict[str, Any], k: int = 5) -> List[Dict[str, Any]]:
        """
        perform multi-query retrieval to find relevant patterns.
        
        args:
            sensor_data: sensor data to analyze
            k: number of neighbors to retrieve
            
        returns:
            retrieved_patterns: list of retrieved patterns with metadata
        """
        # create embedding for the query        
        query_embedding = self._create_embedding(sensor_data)
        #metadata2,text2,meta_keys=util.extract_metadata(sensor_data, self.args)
        #print(f'SENSOR DATA!!!: {sensor_data}')
        # perform retrieval        
        if any('id' in s for s in sensor_data.keys()):#"hostID" in sensor_data:
            
            # first try host-specific search
            matching= [s for s in sensor_data.keys() if 'id' in s]            
            
            host_id = sensor_data.get(matching[0])#sensor_data.get("hostID")
            print(f'GOT ID!!: {matching[0]}:{host_id}\n')
            distances, metadata = self.vector_store.search(
                query_embedding, 
                k=k, 
                filter_func=lambda meta: meta.get(matching[0])==host_id #meta.get('hostID') == host_id
            )
            print(f'RETREIVED DISTANCES!!: {distances}\n')
            print(f'RETREIVED METADATA!!: {metadata}\n')
            
            print('Testing what comes out of the FAISS search:\n')
            test_dist, test_meta=self.vector_store.search(query_embedding, k)
            print(f'RETREIVED TEST DISTANCES!!: {test_dist}\n')
            print(f'RETREIVED TEST METADATA!!: {test_meta}\n')
            
            # if not enough results, do a general search            
            if len([d for d in distances if d != float('inf')]) < k // 2:
                general_distances, general_metadata = self.vector_store.search(query_embedding, k)
                
                # combine results, prioritizing host-specific ones                
                combined_distances = []
                combined_metadata = []
                
                # add host-specific results first                
                for d, m in zip(distances, metadata):
                    if d != float('inf'):
                        combined_distances.append(d)
                        combined_metadata.append(m)
                
                # add general results that aren't already included                
                for d, m in zip(general_distances, general_metadata):
                    if m.get(matching[0]) != host_id:#m.get('hostID') != host_id:
                        combined_distances.append(d)
                        combined_metadata.append(m)
                
                # trim to k results                
                distances = combined_distances[:k]
                metadata = combined_metadata[:k]
        else:
            # general search if no host id            
            distances, metadata = self.vector_store.search(query_embedding, k)
        
        # format retrieved patterns        
        retrieved_patterns = []
        for i, (distance, meta) in enumerate(zip(distances, metadata)):
            if distance == float('inf'):
                continue
            pattern = dict(meta)
            pattern['distance'] = distance
            retrieved_patterns.append(pattern)
        
        return retrieved_patterns
    
    def semantic_pattern_matching(self, 
                                sensor_data: Dict[str, Any], 
                                retrieved_patterns: List[Dict[str, Any]],
                                threshold: float) -> Tuple[bool, Dict[str, Any]]:
        """
        perform semantic pattern matching to detect anomalies.
        
        args:
            sensor_data: sensor data to analyze
            retrieved_patterns: retrieved similar patterns
            threshold: similarity threshold
            
        returns:
            is_anomaly: whether pattern is anomalous based on semantic matching
            metrics: dictionary with semantic matching metrics
        """
        if not retrieved_patterns:
            return True, {"anomaly_reason": "No similar patterns found"}
        
        # compute minimum distance to normal patterns       
        normal_patterns = [p for p in retrieved_patterns if not p.get('is_anomaly', False)]
        min_normal_distance = float('inf')
        closest_normal = None
        
        if normal_patterns:
            min_normal_distance = min(p['distance'] for p in normal_patterns)
            closest_normal = min(normal_patterns, key=lambda p: p['distance'])
        
        # compute minimum distance to anomalous patterns        
        anomaly_patterns = [p for p in retrieved_patterns if p.get('is_anomaly', False)]
        min_anomaly_distance = float('inf')
        closest_anomaly = None
        
        if anomaly_patterns:
            min_anomaly_distance = min(p['distance'] for p in anomaly_patterns)
            closest_anomaly = min(anomaly_patterns, key=lambda p: p['distance'])
        
        # determine if this is an anomaly based on distances        
        is_anomaly = False
        anomaly_reason = ""
        
        if min_normal_distance > threshold:
            is_anomaly = True
            anomaly_reason = f"Distance to closest normal pattern ({min_normal_distance:.4f}) exceeds threshold ({threshold:.4f})"
        elif anomaly_patterns and min_anomaly_distance < min_normal_distance:
            is_anomaly = True
            anomaly_reason = f"More similar to known anomaly ({min_anomaly_distance:.4f}) than to normal pattern ({min_normal_distance:.4f})"
        
        # prepare metrics       
        metrics = {
            "min_normal_distance": min_normal_distance if min_normal_distance != float('inf') else None,
            "closest_normal": closest_normal,
            "min_anomaly_distance": min_anomaly_distance if min_anomaly_distance != float('inf') else None,
            "closest_anomaly": closest_anomaly,
            "anomaly_reason": anomaly_reason
        }
        
        return is_anomaly, metrics
    
    def detect_anomalies(self, 
                        sensor_data: Dict[str, Any],
                        use_llm: bool = True) -> Dict[str, Any]:
        """
        clara: context-aware language-augmented retrieval anomaly detector
        implementation of algorithm 5 from the paper.
        
        args:
            sensor_data: sensor data to analyze
            use_llm: whether to use the llm for analysis
            
        returns:
            result: dictionary containing detection results and explanations
        """
        
        # multi-query retrieval   
        retrieved_patterns = self.multi_query_retrieval(sensor_data)
        
        # sensor data augmentation   
        if use_llm:
            augmented_prompt = self.augmenter.augment_sensor_data(sensor_data, retrieved_patterns)
        
        # semantic pattern matching
        semantic_anomaly, metrics = self.semantic_pattern_matching(
            sensor_data, 
            retrieved_patterns, 
            self.semantic_threshold
        )
        
        # if we can't use the llm, fall back to rule-based analysis        
        if not use_llm:
            return self._rule_based_analysis(sensor_data, retrieved_patterns, 
                                            semantic_anomaly, metrics)
        
        # contextual deviation analysis
        metadata,text2,meta_keys=util.extract_metadata(sensor_data, self.args)
        context={}
        for key, value in sensor_data.items():
            if key in meta_keys['ids'] or key in meta_keys['labels']:
                  context[key]=sensor_data.get(key,'unknown')
        #context = {
        #    "hostID": sensor_data.get("hostID", "unknown"),
        #    "activity": sensor_data.get("activity", "unknown")
        #}
        
        ctx_anomaly_score, ctx_explanation = self.deviation_analyzer.analyze(
            sensor_data, 
            context, 
            retrieved_patterns
        )
        
        # multi-modal pattern recognition      
        multi_modal_anomaly = semantic_anomaly and ctx_anomaly_score > 0.5
        multi_modal_explanation = "Pattern shows unusual relationships between different sensors."
        
        # explanation-driven detection        
        exp_anomaly, exp_confidence, exp_explanation = self.explanation_detector.detect(
            sensor_data, 
            retrieved_patterns
        )
        
        # combine detections    
        # weight the different detection methods        
        combined_anomaly = False
        detection_weights = {
            "semantic": 0.4,
            "contextual": 0.3,
            "multi_modal": 0.1,
            "explanation": 0.2
        }
        
        anomaly_score = (
            semantic_anomaly * detection_weights["semantic"] +
            (ctx_anomaly_score > 0.5) * detection_weights["contextual"] +
            multi_modal_anomaly * detection_weights["multi_modal"] +
            exp_anomaly * detection_weights["explanation"]
        )
        
        if anomaly_score > 0.5:
            combined_anomaly = True
        
        # combine confidences        
        # use explanation confidence as base, adjusted by other methods       
        combined_confidence = exp_confidence
        if semantic_anomaly:
            semantic_confidence = 1.0 - min(1.0, metrics["min_normal_distance"] / (2 * self.semantic_threshold))
            combined_confidence = (combined_confidence + semantic_confidence) / 2
        
        # enhance explanation        
        # combine all explanations, prioritizing the most informative ones        
        enhanced_explanation = exp_explanation
        
        # determine anomaly type         
        anomaly_type = "unknown"
        if combined_anomaly:
            # check if there's a known anomaly type from similar patterns           
            anomaly_patterns = [p for p in retrieved_patterns if p.get('is_anomaly', False)]
            if anomaly_patterns:
                closest_anomaly = min(anomaly_patterns, key=lambda p: p['distance'])
                anomaly_type = closest_anomaly.get('anomaly_type', 'unknown')
        
        # extract more components from the explanation detector result        
        try:
            # try to parse the explanation for structured fields            
            exp_lines = exp_explanation.split('\n\n')
            
            user_friendly_message = ""
            notable_deviations = []
            recommended_actions = []
            
            for section in exp_lines:
                if section.startswith("User-friendly explanation:"):
                    user_friendly_message = section.replace("User-friendly explanation:", "").strip()
                elif section.startswith("Notable deviations:"):
                    deviations_text = section.replace("Notable deviations:", "").strip()
                    notable_deviations = [d.strip('- ') for d in deviations_text.split('\n')]
                elif section.startswith("Recommended actions:"):
                    actions_text = section.replace("Recommended actions:", "").strip()
                    recommended_actions = [a.strip('- ') for a in actions_text.split('\n')]
        
        except Exception:
            # fallback if parsing fails            
            user_friendly_message = "Unusual sensor pattern detected."
            notable_deviations = []
            recommended_actions = []
        
        # prepare result        
        result = {
            "is_anomaly": combined_anomaly,
            "confidence": combined_confidence,
            "anomaly_type": anomaly_type if combined_anomaly else None,
            "explanation": enhanced_explanation,
            "user_friendly_message": user_friendly_message,
            "notable_deviations": notable_deviations,
            "recommended_actions": recommended_actions,
            "similar_patterns": []
                #{
                #    "distance": p.get("distance", float("inf")),
                #    "is_anomaly": p.get("is_anomaly", False),
                #    "activity": p.get("activity", "unknown"),
                #    "description": p.get("description", "") or p.get("explanation", "")
                #}]     
        }
        for p in retrieved_patterns[:3]:# include only top 3 for testing purposes
            #print(f'P!!!!: {p}')            
            sim_pat={}
            for key in p:
                sim_pat[key]=p[key]
            result['similar_patterns'].append(sim_pat)
        return result
    
    def _rule_based_analysis(self, 
                           sensor_data: Dict[str, Any],
                           retrieved_patterns: List[Dict[str, Any]],
                           semantic_anomaly: bool,
                           metrics: Dict[str, Any]) -> Dict[str, Any]:
        """
        perform rule-based anomaly analysis as a fallback.
        
        args:
            sensor_data: sensor data to analyze
            retrieved_patterns: retrieved similar patterns
            semantic_anomaly: whether semantic pattern matching detected an anomaly
            metrics: metrics from semantic pattern matching
            
        returns:
            result: analysis result with anomaly detection and explanation
        """
        # check if there are any similar patterns at all        
        if not retrieved_patterns:
            return {
                "is_anomaly": True,
                "confidence": 0.9,
                "anomaly_type": "unknown",
                "explanation": "No similar patterns found in the database.",
                "similar_patterns": []
            }
        
        # check if any similar patterns are anomalies        
        anomaly_patterns = [p for p in retrieved_patterns if p.get('is_anomaly', False)]
        
        # determine if this is an anomaly based on similarity to known patterns        
        is_anomaly = semantic_anomaly
        anomaly_type = None
        explanation = None
        confidence = 0.5
        
        if semantic_anomaly:
            # no similar patterns found - potential anomaly            
            explanation = metrics.get("anomaly_reason", "Pattern differs significantly from known normal patterns.")
            confidence = min(0.9, metrics.get("min_normal_distance", 1.0) / self.semantic_threshold)
        elif anomaly_patterns:
            # similar to known anomalies            
            is_anomaly = True
            closest_anomaly = min(anomaly_patterns, key=lambda p: p.get('distance', float('inf')))
            anomaly_type = closest_anomaly.get("anomaly_type", "unknown")
            explanation = closest_anomaly.get("explanation", "Similar to known anomaly pattern.")
            confidence = 0.9 - metrics.get("min_anomaly_distance", 0.5)
        else:
            # similar to normal patterns            
            is_anomaly = False
            explanation = "Similar to known normal patterns."
            confidence = 0.9 - metrics.get("min_normal_distance", 0.1)
        
        # prepare result        
        result = {
            "is_anomaly": is_anomaly,
            "confidence": max(0.1, min(0.9, confidence)),
            "anomaly_type": anomaly_type if is_anomaly else None,
            "explanation": explanation,
            "similar_patterns": [
                #{
                #    "distance": p.get("distance", float("inf")),
                #    "is_anomaly": p.get("is_anomaly", False),
                #    "activity": p.get("activity", "unknown"),
                #    "description": p.get("description", "") or p.get("explanation", "")
                #}
                #for p in retrieved_patterns[:3]           
                ]
        }
        for p in retrieved_patterns[:3]:# include only top 3 for testing purposes            
            sim_pat={}
            for key, val in p:
                sim_pat[key]=val
            result['similar_patterns'].append(sim_pat)
        return result
    
    def save(self, vector_store_path: str) -> None:
        """
        save the vector store to disk.
        
        args:
            vector_store_path: path to save the faiss index
        """
        self.vector_store.save(vector_store_path)


if __name__ == "__main__": 
    clara = CLARA()
    
    # create some sample data    
    normal_pattern = {
        "user_id": "user123",
        "activity": "walking",
        "timestamp": "2023-01-01T12:00:00",
        "acc_x": 0.1,
        "acc_y": 9.8,
        "acc_z": 0.2,
        "gyro_x": 0.01,
        "gyro_y": 0.02,
        "gyro_z": 0.01
    }
    
    anomaly_pattern = {
        "user_id": "user123",
        "activity": "walking",
        "timestamp": "2023-01-02T15:30:00",
        "acc_x": 0.5,
        "acc_y": 8.5,
        "acc_z": 0.9,
        "gyro_x": 0.12,
        "gyro_y": 0.22,
        "gyro_z": 0.11
    }
    
    # add patterns to the database    
    clara.add_normal_pattern(normal_pattern, "Normal walking pattern")
    clara.add_anomaly_pattern(
        anomaly_pattern,
        "behavioral",
        "Irregular walking pattern with unusual acceleration"
    )
    
    # test pattern - mix of normal and anomaly    
    test_pattern = {
        "user_id": "user123",
        "activity": "walking",
        "timestamp": "2023-01-03T10:15:00",
        "acc_x": 0.3,
        "acc_y": 9.0,
        "acc_z": 0.5,
        "gyro_x": 0.08,
        "gyro_y": 0.15,
        "gyro_z": 0.07
    }
    
    # detect anomalies    
    print("Testing with LLM:")
    try:
        result = clara.detect_anomalies(test_pattern, use_llm=True)
        print(f"Is anomaly: {result['is_anomaly']}")
        print(f"Confidence: {result['confidence']:.2f}")
        print(f"Anomaly type: {result['anomaly_type']}")
        print(f"Explanation: {result['explanation']}")
    except Exception as e:
        print(f"LLM-based detection failed: {e}")
        print("Falling back to rule-based analysis...")
    
    print("\nTesting with rule-based analysis:")
    result = clara.detect_anomalies(test_pattern, use_llm=False)
    print(f"Is anomaly: {result['is_anomaly']}")
    print(f"Confidence: {result['confidence']:.2f}")
    print(f"Anomaly type: {result['anomaly_type']}")
    print(f"Explanation: {result['explanation']}")