import numpy as np
import faiss
import pickle
import os
import json
from typing import Dict, List, Tuple, Union, Optional, Any

class FAISSVectorStore:

    def __init__(self, 
                 dimension: int = 768, 
                 index_type: str = "flat", 
                 nlist: int = 100, 
                 m: int = 8,
                 use_gpu: bool = False,
                 metadata_file: str = "metadata.pkl"):

        self.dimension = dimension
        self.index_type = index_type
        self.metadata_file = metadata_file
        self.metadata = {}
        self.next_id = 0
        
        # create the appropriate faiss index based on the specified type        
        if index_type == "flat":
            # indexflatl2 for exact nearest neighbor search (development)           
            self.index = faiss.IndexFlatL2(dimension)
        elif index_type == "ivfpq":
            # indexivfpq for approximate nearest neighbor search (production)            
            quantizer = faiss.IndexFlatL2(dimension)
            self.index = faiss.IndexIVFPQ(quantizer, dimension, nlist, m, 8)
            self.index.nprobe = 10  # number of clusters to visit during search        
        else:
            raise ValueError(f"Unsupported index type: {index_type}")
        
        # use gpu if requested and available        
        if use_gpu and faiss.get_num_gpus() > 0:
            self.index = faiss.index_cpu_to_gpu(faiss.StandardGpuResources(), 0, self.index)
            self.using_gpu = True
        else:
            self.using_gpu = False
            
        # load existing metadata if available        
        if os.path.exists(metadata_file):
            with open(metadata_file, 'rb') as f:
                self.metadata = pickle.load(f)
                # find the highest id to continue from                
                if self.metadata:
                    self.next_id = max(int(id) for id in self.metadata.keys()) + 1
    
    def add_embedding(self, 
                      embedding: np.ndarray, 
                      metadata: Dict[str, Any]) -> int:
        # ensure the embedding is in the right format        
        if len(embedding.shape) == 1:
            embedding = embedding.reshape(1, -1)
        
        # ensure the embedding has the correct dimension       
        if embedding.shape[1] != self.dimension:
            raise ValueError(f"Embedding dimension {embedding.shape[1]} does not match index dimension {self.dimension}")
        
        # convert to float32 if needed        
        if embedding.dtype != np.float32:
            embedding = embedding.astype(np.float32)
        
        # add to the index        
        if self.index_type == "ivfpq" and not self.index.is_trained:
            # for ivf indexes, we need to train before adding the first vector            
            # in a real system, we'd train on a larger representative set            
            self.index.train(embedding)
        
        # add the vector to the index        
        self.index.add(embedding)
        
        # store metadata        
        embedding_id = str(self.next_id)
        self.metadata[embedding_id] = metadata
        self.next_id += 1
        
        # save metadata to disk        
        with open(self.metadata_file, 'wb') as f:
            pickle.dump(self.metadata, f)
        
        return int(embedding_id)
    
    def add_embeddings(self, 
                       embeddings: np.ndarray, 
                       metadata_list: List[Dict[str, Any]]) -> List[int]:

        if len(embeddings) != len(metadata_list):
            raise ValueError("Number of embeddings and metadata entries must match")
        
        # ensure embeddings are in the right format        
        if embeddings.shape[1] != self.dimension:
            raise ValueError(f"Embedding dimension {embeddings.shape[1]} does not match index dimension {self.dimension}")
        
        # convert to float32 if needed        
        if embeddings.dtype != np.float32:
            embeddings = embeddings.astype(np.float32)
        
        # add to the index        
        if self.index_type == "ivfpq" and not self.index.is_trained:
            self.index.train(embeddings)
        
        self.index.add(embeddings)
        
        # store metadata        
        ids = []
        for metadata in metadata_list:
            embedding_id = str(self.next_id)
            self.metadata[embedding_id] = metadata
            ids.append(int(embedding_id))
            self.next_id += 1
        
        # save metadata to disk        
        with open(self.metadata_file, 'wb') as f:
            pickle.dump(self.metadata, f)
        
        return ids
    
    def search(self, 
               query_embedding: np.ndarray, 
               k: int = 5, 
               filter_func: Optional[callable] = None) -> Tuple[List[float], List[Dict[str, Any]]]:
        
        # ensure the query is in the right format        
        if len(query_embedding.shape) == 1:
            query_embedding = query_embedding.reshape(1, -1)
        
        # ensure the query has the correct dimension        
        if query_embedding.shape[1] != self.dimension:
            raise ValueError(f"Query dimension {query_embedding.shape[1]} does not match index dimension {self.dimension}")
        
        # convert to float32 if needed        
        if query_embedding.dtype != np.float32:
            query_embedding = query_embedding.astype(np.float32)
        
        # perform the search        
        distances, indices = self.index.search(query_embedding, k)
        
        # extract metadata for the results        
        result_distances = distances[0].tolist()
        result_metadata = []
        
        for idx in indices[0]:
            if idx != -1:  # faiss returns -1 for padded results if fewer than k are found                
                metadata = self.metadata.get(str(idx), {})
                if filter_func is None or filter_func(metadata):
                    result_metadata.append(metadata)
            else:
                result_metadata.append({})
        
        return result_distances, result_metadata
    
    def search_by_user(self, 
                       query_embedding: np.ndarray, 
                       hostID: str, 
                       k: int = 5) -> Tuple[List[float], List[Dict[str, Any]]]:
        return self.search(
            query_embedding, 
            k, 
            filter_func=lambda metadata: metadata.get('hostID') == hostID
        )
    
    def search_by_activity(self, 
                          query_embedding: np.ndarray, 
                          activity: str, 
                          k: int = 5) -> Tuple[List[float], List[Dict[str, Any]]]:
        return self.search(
            query_embedding, 
            k, 
            filter_func=lambda metadata: metadata.get('activity') == activity
        )
    
    def search_anomalies(self, 
                         query_embedding: np.ndarray, 
                         k: int = 5) -> Tuple[List[float], List[Dict[str, Any]]]:
        return self.search(
            query_embedding, 
            k, 
            filter_func=lambda metadata: metadata.get('is_anomaly', False) == True
        )
    
    def save(self, index_file: str) -> None:
        if self.using_gpu:
            # convert back to cpu for saving            
            cpu_index = faiss.index_gpu_to_cpu(self.index)
            faiss.write_index(cpu_index, index_file)
        else:
            faiss.write_index(self.index, index_file)
    
    def load(self, index_file: str) -> None:
        self.index = faiss.read_index(index_file)
        
        # move to gpu if needed        
        if self.using_gpu:
            self.index = faiss.index_cpu_to_gpu(faiss.StandardGpuResources(), 0, self.index)
    
    def get_stats(self) -> Dict[str, Any]:
        stats = {
            "total_vectors": self.index.ntotal,
            "dimension": self.dimension,
            "index_type": self.index_type,
            "using_gpu": self.using_gpu,
            "metadata_count": len(self.metadata),
            "is_trained": getattr(self.index, "is_trained", True)
        }
        
        return stats


# this is just an example usage for testing and demo purposes
if __name__ == "__main__":
    # create a vector store with 768-dimensional embeddings    
    vector_store = FAISSVectorStore(dimension=768, index_type="flat")
    
    # add some example embeddings   
    normal_pattern = np.random.random((1, 768)).astype(np.float32)
    normal_metadata = {
        "hostID": "user123",
        "activity": "walking",
        "timestamp": "2023-01-01T12:00:00",
        "is_anomaly": False,
        "description": "Normal walking pattern with regular gait and consistent pace"
    }
    
    vector_store.add_embedding(normal_pattern, normal_metadata)
    
    # add an anomalous pattern    
    anomaly_pattern = np.random.random((1, 768)).astype(np.float32)
    anomaly_metadata = {
        "hostID": "user123",
        "activity": "walking",
        "timestamp": "2023-01-02T15:30:00",
        "is_anomaly": True,
        "description": "Irregular walking pattern with sudden stops and uneven pace",
        "anomaly_type": "behavioral",
        "explanation": "The pattern shows unusual pauses and acceleration changes that deviate from the user's normal walking behavior."
    }
    
    vector_store.add_embedding(anomaly_pattern, anomaly_metadata)
    
    # search for similar patterns    
    query = np.random.random((1, 768)).astype(np.float32)
    distances, metadata = vector_store.search(query, k=2)
    
    print("Search results:")
    for i, (distance, meta) in enumerate(zip(distances, metadata)):
        print(f"Result {i+1}: Distance = {distance}")
        print(f"  User: {meta.get('hostID')}")
        print(f"  Activity: {meta.get('activity')}")
        print(f"  Is Anomaly: {meta.get('is_anomaly')}")
        print(f"  Description: {meta.get('description')}")
        print()
    
    # save the index    
    vector_store.save("example_index.faiss")
    
    # print statistics    
    stats = vector_store.get_stats()
    print("Vector Store Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
