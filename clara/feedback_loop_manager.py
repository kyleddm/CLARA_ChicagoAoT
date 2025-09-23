import json
import os
import time
import datetime
import numpy as np
from typing import Dict, List, Any, Optional, Tuple

class FeedbackLoopManager:
    
    def __init__(self, 
                detector, 
                feedback_log_path: str = "feedback_log.json",
                auto_adapt: bool = True):

        self.detector = detector
        self.feedback_log_path = feedback_log_path
        self.auto_adapt = auto_adapt # trying to see the limitation of this without expert's manual correction. 
        self.feedback_log = self._load_feedback_log()
        
        # stats for feedback incorporation
        self.stats = {
            "total_feedback_entries": 0,
            "false_positives": 0,
            "false_negatives": 0,
            "confirmed_anomalies": 0,
            "confirmed_normals": 0,
            "last_update": None
        }
    
    def _load_feedback_log(self) -> List[Dict[str, Any]]:

        if os.path.exists(self.feedback_log_path):
            try:
                with open(self.feedback_log_path, 'r') as f:
                    return json.load(f)
            except json.JSONDecodeError:
                print(f"Error loading feedback log, creating new one")
                return []
        return []
    
    def _save_feedback_log(self) -> None:

        with open(self.feedback_log_path, 'w') as f:
            json.dump(self.feedback_log, f, indent=2)
    
    def add_feedback(self, 
                    sensor_data: Dict[str, Any], 
                    detection_result: Dict[str, Any],
                    user_feedback: Dict[str, Any]) -> Dict[str, Any]:

        # create feedback entry
        timestamp = time.strftime("%Y-%m-%dT%H:%M:%S")
        
        feedback_entry = {
            "timestamp": timestamp,
            "sensor_data": sensor_data,
            "detection_result": detection_result,
            "user_feedback": user_feedback,
            "processed": False
        }
        
        # add to log
        self.feedback_log.append(feedback_entry)
        self._save_feedback_log()
        
        # update stats
        self.stats["total_feedback_entries"] += 1
        
        was_anomaly = detection_result.get("is_anomaly", False)
        should_be_anomaly = user_feedback.get("actual_anomaly", False)
        
        if was_anomaly and not should_be_anomaly:
            self.stats["false_positives"] += 1
        elif not was_anomaly and should_be_anomaly:
            self.stats["false_negatives"] += 1
        elif was_anomaly and should_be_anomaly:
            self.stats["confirmed_anomalies"] += 1
        else:  # not was_anomaly and not should_be_anomaly
            self.stats["confirmed_normals"] += 1
        
        # process feedback immediately if auto_adapt is enabled
        if self.auto_adapt:
            self.process_feedback(feedback_entry)
        
        return feedback_entry
    
    def process_feedback(self, feedback_entry: Dict[str, Any]) -> bool:

        if feedback_entry.get("processed", False):
            return False  # already processed
        
        sensor_data = feedback_entry.get("sensor_data", {})
        user_feedback = feedback_entry.get("user_feedback", {})
        
        # check if we have the minimum required info
        if not sensor_data or "actual_anomaly" not in user_feedback:
            return False
        
        # update the knowledge base based on feedback
        is_actually_anomaly = user_feedback.get("actual_anomaly", False)
        
        try:
            if is_actually_anomaly:
                # add to known anomalies
                anomaly_type = user_feedback.get("anomaly_type", "unknown")
                explanation = user_feedback.get("notes", "Anomaly identified through user feedback")
                
                self.detector.add_anomaly_pattern(
                    sensor_data,
                    anomaly_type=anomaly_type,
                    explanation=explanation
                )
            else:
                # add to normal patterns
                description = user_feedback.get("notes", "Normal pattern confirmed through user feedback")
                
                self.detector.add_normal_pattern(
                    sensor_data,
                    description=description
                )
            
            # mark as processed
            feedback_entry["processed"] = True
            self._save_feedback_log()
            
            # update stats
            self.stats["last_update"] = time.strftime("%Y-%m-%dT%H:%M:%S")
            
            return True
        
        except Exception as e:
            print(f"Error processing feedback: {e}")
            return False
    
    def process_all_feedback(self) -> Dict[str, Any]:

        results = {
            "total": 0,
            "processed": 0,
            "failed": 0
        }
        
        for entry in self.feedback_log:
            if not entry.get("processed", False):
                results["total"] += 1
                
                if self.process_feedback(entry):
                    results["processed"] += 1
                else:
                    results["failed"] += 1
        
        return results
    
    def get_feedback_stats(self) -> Dict[str, Any]:

        # update total count
        self.stats["total_feedback_entries"] = len(self.feedback_log)
        
        # calculate percentages
        total = self.stats["total_feedback_entries"]
        if total > 0:
            self.stats["false_positive_rate"] = (self.stats["false_positives"] / total) * 100
            self.stats["false_negative_rate"] = (self.stats["false_negatives"] / total) * 100
            self.stats["accuracy"] = ((self.stats["confirmed_anomalies"] + self.stats["confirmed_normals"]) / total) * 100
        else:
            self.stats["false_positive_rate"] = 0
            self.stats["false_negative_rate"] = 0
            self.stats["accuracy"] = 0
        
        return self.stats
    
    def get_recent_feedback(self, limit: int = 10) -> List[Dict[str, Any]]:

        # sort by timestamp (newest first) and return the limited number
        sorted_entries = sorted(
            self.feedback_log, 
            key=lambda x: x.get("timestamp", ""), 
            reverse=True
        )
        
        return sorted_entries[:limit]
    
    def save_knowledgebase(self, vector_store_path: str) -> bool:

        try:
            self.detector.save(vector_store_path)
            return True
        except Exception as e:
            print(f"Error saving knowledge base: {e}")
            return False
    
    def clear_feedback_log(self, backup: bool = True) -> bool:

        try:
            if backup and os.path.exists(self.feedback_log_path):
                backup_path = f"{self.feedback_log_path}.{time.strftime('%Y%m%d%H%M%S')}.bak"
                with open(self.feedback_log_path, 'r') as src, open(backup_path, 'w') as dst:
                    dst.write(src.read())
            
            self.feedback_log = []
            self._save_feedback_log()
            
            # reset stats
            self.stats = {
                "total_feedback_entries": 0,
                "false_positives": 0,
                "false_negatives": 0,
                "confirmed_anomalies": 0,
                "confirmed_normals": 0,
                "last_update": time.strftime("%Y-%m-%dT%H:%M:%S")
            }
            
            return True
        except Exception as e:
            print(f"Error clearing feedback log: {e}")
            return False


# example of how to use the feedback loop
if __name__ == "__main__":
    from rag_anomaly_detection_updated import RAGAnomalyDetector
    
    # initialize the detector
    detector = RAGAnomalyDetector()
    
    # initialize the feedback loop
    feedback_loop = FeedbackLoopManager(detector)
    
    # example sensor data and detection result
    sensor_data = {
        "hostID": "user123",
        "activity": "walking",
        "timestamp": "2023-01-01T12:00:00",
        "acc_x": 0.1,
        "acc_y": 9.8,
        "acc_z": 0.2,
        "gyro_x": 0.01,
        "gyro_y": 0.02,
        "gyro_z": 0.01
    }
    
    detection_result = {
        "is_anomaly": True,
        "confidence": 0.8,
        "anomaly_type": "behavioral",
        "explanation": "The pattern shows unusual values in accelerometer and gyroscope readings."
    }
    
    # simulate user feedback (disagreeing with the detection)
    user_feedback = {
        "correct": False,
        "actual_anomaly": False,
        "notes": "This is actually my normal walking pattern."
    }
    
    # add feedback
    feedback_entry = feedback_loop.add_feedback(sensor_data, detection_result, user_feedback)
    print(f"Added feedback: {feedback_entry['timestamp']}")
    
    # process all feedback
    results = feedback_loop.process_all_feedback()
    print(f"Processed feedback: {results}")
    
    # get feedback stats
    stats = feedback_loop.get_feedback_stats()
    print(f"Feedback stats: {stats}")
    
    # save the updated knowledge base
    feedback_loop.save_knowledgebase("updated_vector_store.faiss")