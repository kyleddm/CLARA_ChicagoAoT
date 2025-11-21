import os
import sys
import time
import argparse
import json
import torch
import numpy as np
from typing import Dict, List, Any, Optional

from clara.clara_detector import CLARA
from clara.feedback_loop_manager import FeedbackLoopManager
from clara.extrasensory_csv_loader import ExtraSensoryCSVLoader
from aot.aot_csv_loader import AotCSVLoader
from aot.contrastive_embedding_model import generate_sensor_embeddings_with_contrastive_learning
from clara.sensor_data_augmenter import SensorDataAugmenter
from clara.contextual_deviation_analyzer import ContextualDeviationAnalyzer
from clara.explanation_driven_detector import ExplanationDrivenDetector
from utilities import pruneTime

# constants
DEFAULT_CSV_PATH = "/mnt/e/input/clara/no_watch_data_imputed_replaced_cleaned.csv"#"/home/ai-lab2/GAIN-Pytorch-master/data/no_watch_data_imputed_replaced_cleaned.csv"
DEFAULT_MODEL = "llama3.2:1b"
DEFAULT_API_BASE = "http://localhost:11434"
DEFAULT_VECTOR_STORE = "extrasensory_vector_store.faiss"
DEFAULT_FEEDBACK_LOG = "feedback_log.json"


def initialize_detector(args):
    
    print(f"Initializing CLARA anomaly detector with {args.model}...")
    
    # use a smaller embedding dimension for better performance (make sure to change this if you want to constrain your sensor data in a different way)
    embedding_dim = args.embedding_dim
    
    # create the detector    
    detector = CLARA(
        vector_store_path=args.vector_store if os.path.exists(args.vector_store) else None,
        llm_model_name=args.model,
        llm_api_base=args.api_base,
        embedding_dim=embedding_dim,
        semantic_threshold=args.threshold,
        coherence_threshold=args.coherence
    )
    
    return detector


def initialize_feedback_loop(detector, args):
    
    print(f"Initializing feedback loop manager...")
    
    # create the feedback loop manager    
    feedback_loop = FeedbackLoopManager(
        detector=detector,
        feedback_log_path=args.feedback_log,
        auto_adapt=args.auto_adapt
    )
    
    return feedback_loop


def load_training_data(detector, args):
    
    print(f"Loading training data from {args.csv_path}...")
    
    # load dataset    
    if args.skip_training:
        print("Skipping training data loading as requested.")
        return
    
    # use csv loader   #this needs to be generalized to load any csv data KDM 2025NOV21 
    csv_loader = AotCSVLoader(args.csv_path)
    
    # get available users    
    nodes = csv_loader.get_available_nodes()
    if not nodes:
        print("No users found in the dataset. Using synthetic data.")
        data = {"synthetic_node": csv_loader.generate_synthetic_data(args.max_samples)}
    else:
        data = {}
        for node_id in nodes[:3]:  # limit to first 3 nodes.  why?  KDM 2025NOV21            
            node_data = csv_loader.load_node_data(node_id, max_samples=args.max_samples)
            if node_data:
                data[node_id] = node_data
                print(f"Loaded {len(node_data)} samples for user {node_id}")
        
        if not data:
            print("No valid data found in the dataset. Using synthetic data.")
            data = {"synthetic_node": csv_loader.generate_synthetic_data(args.max_samples)}
    
    # count total samples    
    total_samples = sum(len(samples) for samples in data.values())
    print(f"Loaded {total_samples} samples across {len(data)} nodes")
    
    # train embedding model using contrastive learning if enough data    
    if total_samples >= 10 and args.train_embeddings:
        print("Training contrastive embedding model...")
        try:
            # prepare data for embedding training            
            all_samples = []
            for node_samples in data.values(): #this gives us the dictionaries for each node containing the other data columns
                all_samples.extend(node_samples)
            
            # extract features and labels            
            features = []
            labels = []
            for sample in all_samples:
                #extract numerical features.  Time (without the year) is used here since it's very relevant
                feature_vector=[pruneTime(sample['timestamp']),sample['value_hrf']]
                if feature_vector:
                    features.append(feature_vector)
                    #we're going to use subsystem, sensor, and parameter labels as relevant data for the feature, but these need to be made into embeddings
                    subsys_sen=hash((sample['subsystem']+'_'+sample['sensor']) % 100)
                    #we combine the subsystem (make) and sensor (model) together because what matters is the similarity between srnsors of similar parameters; we don't want CLARA adding the same weight to two temp sensors of different models and two completely different sensors of similar make
                    #sen=hash(sample['sensor'] % 100)
                    param=hash(sample['parameter'] % 100)
                    labels.append([subsys_sen,param])
                    
                    
            #for sample in all_samples:
            #    # extract numerical features                
            #    feature_vector = []
            #    for key, value in sorted(sample.items()):
            #        if key not in ['node_id', 'timestamp', 'subsystem' ,'sensor', 'parameter'] and isinstance(value, (int, float)):
            #            feature_vector.append(float(value))
            #    
            #    if feature_vector:
            #        features.append(feature_vector)
            #        # use activity as label                    
            #        activity = sample.get('activity', 'unknown')
            #        labels.append(hash(activity) % 100)  # simple hash to convert activity to numeric label            
            # pad features to same length if needed            
            max_length = max(len(f) for f in features)
            padded_features = []
            for f in features:
                if len(f) < max_length:
                    padded_features.append(f + [0.0] * (max_length - len(f)))
                else:
                    padded_features.append(f)
            
            features_array = np.array(padded_features, dtype=np.float32)
            #Note that the labels at this point are in the wrong direction;  each sample produces the list of labels and that is appended to the labels array.  we need to transpose it!
            transposed = [list(row) for row in zip(*labels)]
            labels_array = np.array(transposed, dtype=np.int64)
            
            # train embedding model            
            model, embeddings = generate_sensor_embeddings_with_contrastive_learning(
                feature_vectors=features_array,
                labels=labels_array,
                embedding_dim=768,
                epochs=10  # fewer epochs for demonstration           
                )
            
            # use the trained model            
            detector.embedding_model = model
            print("Successfully trained embedding model!")
        except Exception as e:
            print(f"Error training embedding model: {e}")
            print("Using default embedding approach instead.")
    
    # add to vector store    
    patterns_added = 0
    anomalies_added = 0
    
    for user_id, samples in data.items():
        print(f"Processing data for user {user_id}...")
        
        # reserve some samples for anomalies (approximately 10%)        
        num_anomalies = max(1, int(len(samples) * 0.1))
        normal_samples = samples[:-num_anomalies]
        anomaly_samples = samples[-num_anomalies:]
        
        # add normal patterns        
        for i, sample in enumerate(normal_samples):
            if i % 10 == 0:
                print(f"  Added {i}/{len(normal_samples)} normal patterns...", end="\r")
            
            detector.add_normal_pattern(
                sample,
                description=f"Normal pattern for user {user_id} while {sample.get('activity', 'unknown activity')}"
            )
            patterns_added += 1
        
        print(f"  Added {len(normal_samples)}/{len(normal_samples)} normal patterns.")
        
        # add anomaly patterns        
        for i, sample in enumerate(anomaly_samples):
            # modify samples to make them more anomalous            
            for key in sample:
                if key not in ['user_id', 'activity', 'timestamp'] and isinstance(sample[key], (int, float)):
                    # randomly increase or decrease by a factor                    
                    if i % 2 == 0:
                        sample[key] = sample[key] * 2.5
                    else:
                        sample[key] = sample[key] * 0.4
            
            detector.add_anomaly_pattern(
                sample,
                anomaly_type="behavioral",
                explanation=f"Simulated anomaly for user {user_id} while {sample.get('activity', 'unknown activity')}"
            )
            anomalies_added += 1
    
    print(f"Added {patterns_added} normal patterns and {anomalies_added} anomalies to the vector store")
    
    # save the vector store    
    detector.save(args.vector_store)
    print(f"Saved vector store to {args.vector_store}")


def run_detection_demo(detector, feedback_loop, args):
    
    print("\nRunning anomaly detection demonstration...")
    
    # load test data (a small subset)    
    csv_loader = ExtraSensoryCSVLoader(args.csv_path)
    users = csv_loader.get_available_users()
    
    if not users:
        print("No users found in the dataset. Using synthetic test data.")
        test_data = csv_loader.generate_synthetic_data(5)
    else:
        # get data for the first user        
        user_id = users[0]
        raw_test_data = csv_loader.load_user_data(user_id, max_samples=5)
        
        if not raw_test_data:
            print(f"No data found for user {user_id}. Using synthetic test data.")
            test_data = csv_loader.generate_synthetic_data(5)
        else:
            test_data = raw_test_data
            
            # create an anomalous sample by modifying the last sample            
            if len(test_data) > 0:
                anomaly_sample = test_data[-1].copy()
                # modify several sensor values to make it anomalous                
                for key in anomaly_sample:
                    if key not in ['user_id', 'activity', 'timestamp'] and isinstance(anomaly_sample[key], (int, float)):
                        anomaly_sample[key] = anomaly_sample[key] * 3.0
                
                test_data.append(anomaly_sample)
    
    print(f"Running detection on {len(test_data)} test samples...")
    
    for i, sample in enumerate(test_data):
        print(f"\nSample {i+1}/{len(test_data)}:")
        print(f"User: {sample.get('user_id', 'unknown')}, Activity: {sample.get('activity', 'unknown')}")
        
        # run detection        
        try:
            start_time = time.time()
            result = detector.detect_anomalies(sample, use_llm=True)
            end_time = time.time()
            
            # print results            
            is_anomaly = result.get("is_anomaly", False)
            confidence = result.get("confidence", 0.0)
            
            print(f"Detection result: {'ANOMALY' if is_anomaly else 'NORMAL'} (confidence: {confidence:.2f})")
            print(f"Detection time: {end_time - start_time:.2f} seconds")
            
            # print human-friendly explanation            
            if "user_friendly_message" in result:
                print("\nExplanation:")
                print(f"  {result['user_friendly_message']}")
            else:
                print("\nExplanation:")
                print(f"  {result.get('explanation', 'No explanation available')}")
            
            # if anomaly, print additional details            
            if is_anomaly and "notable_deviations" in result:
                print("\nNotable deviations:")
                for deviation in result.get("notable_deviations", []):
                    print(f"  - {deviation}")
            
            # simulate user feedback    
                    
            # for demonstration, I gave an assumption that every other sample is correctly classified            
            correct = (i % 2 == 0)
            # last sample is always an anomaly            
            actual_anomaly = (i == len(test_data) - 1)
            
            user_feedback = {
                "correct": correct,
                "actual_anomaly": actual_anomaly,
                "notes": f"{'Correct classification' if correct else 'Incorrect classification'}"
            }
            
            # add feedback            
            feedback_entry = feedback_loop.add_feedback(sample, result, user_feedback)
            print(f"\nAdded user feedback: {user_feedback['correct']} (ID: {feedback_entry['timestamp']})")
            
        except Exception as e:
            print(f"Error during detection: {e}")
            print("Trying rule-based detection instead...")
            
            try:
                result = detector.detect_anomalies(sample, use_llm=False)
                print(f"Rule-based result: {'ANOMALY' if result.get('is_anomaly', False) else 'NORMAL'}")
            except Exception as e2:
                print(f"Rule-based detection also failed: {e2}")
    
    # process feedback    
    print("\nProcessing collected feedback...")
    results = feedback_loop.process_all_feedback()
    print(f"Processed {results['processed']}/{results['total']} feedback entries")
    
    # get feedback stats    
    stats = feedback_loop.get_feedback_stats()
    print("\nFeedback statistics:")
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.2f}")
        else:
            print(f"  {key}: {value}")
    
    # save updated vector store    
    detector.save(args.vector_store)
    print(f"\nSaved updated vector store to {args.vector_store}")
    
    return test_data[-1]  # return the last (anomalous) sample for detailed analysis

def analyze_sample_in_detail(detector, sample, args):
    
    print("\nPerforming detailed analysis of sample...")
    
    try:
        # 1. retrieve similar patterns        
        retrieved_patterns = detector.multi_query_retrieval(sample)
        print(f"Found {len(retrieved_patterns)} similar patterns")
        
        # 2. perform semantic pattern matching        
        semantic_anomaly, metrics = detector.semantic_pattern_matching(
            sample, retrieved_patterns, detector.semantic_threshold
        )
        print(f"\nSemantic Pattern Matching:")
        print(f"  Is Anomaly: {semantic_anomaly}")
        print(f"  Reason: {metrics.get('anomaly_reason', 'None')}")
        
        # 3. only proceed with llm-based analysis if available        
        if args.skip_llm:
            print("\nSkipping LLM analysis as requested.")
            return
        
        # 4. perform sensor data augmentation        
        augmenter = SensorDataAugmenter()
        augmented_prompt = augmenter.augment_sensor_data(sample, retrieved_patterns)
        print("\nAugmented Sensor Data (preview):")
        print(augmented_prompt[:200] + "...")
        
        # 5. perform contextual deviation analysis        
        context = {
            "user_id": sample.get("user_id", "unknown"),
            "activity": sample.get("activity", "unknown")
        }
        deviation_analyzer = ContextualDeviationAnalyzer(detector.llm)
        try:
            ctx_anomaly_score, ctx_explanation = deviation_analyzer.analyze(
                sample, context, retrieved_patterns
            )
            print(f"\nContextual Deviation Analysis:")
            print(f"  Anomaly Score: {ctx_anomaly_score:.4f}")
            print(f"  Explanation: {ctx_explanation[:100]}...")
        except Exception as e:
            print(f"Error in contextual deviation analysis: {e}")
        
        # 6. perform explanation-driven detection        
        explanation_detector = ExplanationDrivenDetector(
            detector.llm, coherence_threshold=detector.coherence_threshold
        )
        try:
            exp_anomaly, exp_confidence, exp_explanation = explanation_detector.detect(
                sample, retrieved_patterns
            )
            print(f"\nExplanation-Driven Detection:")
            print(f"  Is Anomaly: {exp_anomaly}")
            print(f"  Confidence: {exp_confidence:.4f}")
            print(f"  Explanation: {exp_explanation[:100]}...")
        except Exception as e:
            print(f"Error in explanation-driven detection: {e}")
        
        # 7. run full clara detection        
        result = detector.detect_anomalies(sample, use_llm=True)
        
        # print full json result        
        print("\nFull CLARA analysis result:")
        print(json.dumps(result, indent=2))
        
    except Exception as e:
        print(f"Error during detailed analysis: {e}")


def parse_arguments():

    parser = argparse.ArgumentParser(description="CLARA: Context-aware Language-Augmented Retrieval Anomaly Detection with Llama-3.2-1B")
    
    # data and model paths    
    parser.add_argument("--csv-path", default=DEFAULT_CSV_PATH,
                        help=f"Path to ExtraSensory CSV file (default: {DEFAULT_CSV_PATH})")
    parser.add_argument("--model", default=DEFAULT_MODEL,
                        help=f"Name of Llama model in Ollama (default: {DEFAULT_MODEL})")
    parser.add_argument("--api-base", default=DEFAULT_API_BASE,
                        help=f"Base URL for Ollama API (default: {DEFAULT_API_BASE})")
    parser.add_argument("--vector-store", default=DEFAULT_VECTOR_STORE,
                        help=f"Path to FAISS vector store (default: {DEFAULT_VECTOR_STORE})")
    parser.add_argument("--feedback-log", default=DEFAULT_FEEDBACK_LOG,
                        help=f"Path to feedback log file (default: {DEFAULT_FEEDBACK_LOG})")
    
    # algorithm parameters    
    parser.add_argument("--threshold", type=float, default=0.7,
                       help="Threshold for semantic pattern matching (default: 0.7)")
    parser.add_argument("--coherence", type=float, default=0.6,
                       help="Threshold for explanation coherence (default: 0.6)")
    parser.add_argument("--embedding-dim", type=int, default=64,
                       help="Dimension of embedding vectors (default: 64)")
    
    # behavior options    
    parser.add_argument("--skip-training", action="store_true",
                        help="Skip loading training data (use existing vector store)")
    parser.add_argument("--train-embeddings", action="store_true",
                        help="Train contrastive embedding model on data")
    parser.add_argument("--auto-adapt", action="store_true",
                        help="Automatically apply feedback to update the knowledge base")
    parser.add_argument("--max-samples", type=int, default=50,
                        help="Maximum number of samples to load per user (default: 50)")
    parser.add_argument("--skip-llm", action="store_true",
                        help="Skip LLM-based analysis in detailed analysis")
    
    # operations    
    parser.add_argument("--reset-feedback", action="store_true",
                        help="Reset the feedback log before starting")
    
    args = parser.parse_args()
    return args


def main():

    # parse arguments    
    args = parse_arguments()
    
    print("=" * 70)
    print("CLARA: CONTEXT-AWARE LANGUAGE-AUGMENTED RETRIEVAL ANOMALY DETECTION")
    print("=" * 70)
    
    # initialize components    
    detector = initialize_detector(args)
    feedback_loop = initialize_feedback_loop(detector, args)
    
    # reset feedback log if requested    
    if args.reset_feedback:
        print("Resetting feedback log...")
        feedback_loop.clear_feedback_log()
    
    # load training data    
    load_training_data(detector, args)
    
    # run demonstration    
    anomalous_sample = run_detection_demo(detector, feedback_loop, args)
    
    # detailed analysis of a sample    
    analyze_sample_in_detail(detector, anomalous_sample, args)
    
    print("\nDemonstration complete!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
