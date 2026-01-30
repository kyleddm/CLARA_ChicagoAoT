import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple, Union, Optional, Any
import random
from utilities import closest_divisor

class SensorDataset(Dataset):

    def __init__(self, features: np.ndarray, labels: np.ndarray):
        
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


class ContrastiveEmbeddingModel(nn.Module):
    
    def __init__(self, input_dim: int, embedding_dim: int = 768):

        super(ContrastiveEmbeddingModel, self).__init__()
        
        # multi-layer network        
        self.network = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, embedding_dim),
            nn.LayerNorm(embedding_dim)
        )
    
    def forward(self, x):

        return self.network(x)


class ContrastiveLoss(nn.Module):
    
    def __init__(self, margin: float = 1.0, lambda_neg: float = 0.5):

        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.lambda_neg = lambda_neg
    
    def forward(self, embeddings, labels):
 
        batch_size = embeddings.size(0)
        
        # compute pairwise distances        
        dist_matrix = torch.cdist(embeddings, embeddings, p=2)
        
        # create mask for positive pairs (same label)    Note that there are multiple labels so the matrices need to be shifted!    
        pos_mask = (labels.unsqueeze(1) == labels.unsqueeze(2)).float()
        for i in range(pos_mask.size(0)):
            torch.Tensor.fill_diagonal_(pos_mask[i], 0)# exclude self-pairs 
            #pos_mask.fill_diagonal_(0)  # exclude self-pairs        
        # create mask for negative pairs (different label)        
        neg_mask = (labels.unsqueeze(1) != labels.unsqueeze(2)).float()
        
        
        #loss computation: note this sums up over ALL LABELS
        # compute positive pair loss: pull similar patterns together        
        pos_loss = (dist_matrix * pos_mask).sum() / (pos_mask.sum() + 1e-8)
        
        # compute negative pair loss: push dissimilar patterns apart        
        neg_loss = torch.clamp(self.margin - dist_matrix, min=0) * neg_mask
        neg_loss = neg_loss.sum() / (neg_mask.sum() + 1e-8)
        
        # combined loss        
        loss = pos_loss + self.lambda_neg * neg_loss
        
        return loss


def train_embedding_model(features: np.ndarray, 
                         labels: np.ndarray,
                         embedding_dim: int = 768,
                         batch_size: int = 64,
                         epochs: int = 50,
                         learning_rate: float = 0.001,
                         margin: float = 1.0,
                         lambda_neg: float = 0.5,
                         device: str = None) -> Tuple[ContrastiveEmbeddingModel, np.ndarray]:
    """
    train a contrastive embedding model as described in algorithm 1.
    
    args:
        features: array of feature vectors
        labels: array of labels (e.g., activity or user ids)
        embedding_dim: dimensionality of the output embeddings
        batch_size: training batch size
        epochs: number of training epochs
        learning_rate: learning rate for optimization
        margin: margin for negative pairs in contrastive loss
        lambda_neg: weight for negative pair loss
        device: device to use for training ('cpu' or 'cuda')
        
    returns:
        model: trained embedding model
        embeddings: array of embeddings for the input features
    """
   
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    
    # create dataset and data loader
    dataset = SensorDataset(features, labels)
    print(f'SATASET SUZE!!: {len(dataset)}\n')
    batch_size=closest_divisor(len(dataset),batch_size)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # initialize model    
    input_dim = features.shape[1]
    model = ContrastiveEmbeddingModel(input_dim, embedding_dim).to(device)
    
    # initialize loss function and optimizer    
    criterion = ContrastiveLoss(margin=margin, lambda_neg=lambda_neg)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # training loop    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        
        for batch_features, batch_labels in dataloader:          
            batch_features = batch_features.to(device)
            batch_labels = batch_labels.to(device)
                    
            optimizer.zero_grad()
            batch_embeddings = model(batch_features)
            
            loss = criterion(batch_embeddings, batch_labels)
            
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        # debug      
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss / len(dataloader):.4f}")
    
    # final embeddings    
    model.eval()
    with torch.no_grad():
        all_features = torch.tensor(features, dtype=torch.float32).to(device)
        all_embeddings = model(all_features).cpu().numpy()
    
    return model, all_embeddings


def generate_sensor_embeddings_with_contrastive_learning(
    feature_vectors: np.ndarray,
    labels: np.ndarray,
    embedding_dim: int = 768,
    batch_size: int = 64,
    epochs: int = 50,
    margin: float = 1.0,
    lambda_neg: float = 0.5
    ) -> Tuple[ContrastiveEmbeddingModel, np.ndarray]:
    
    model, embeddings = train_embedding_model(
        features=feature_vectors,
        labels=labels,
        embedding_dim=embedding_dim,
        batch_size=batch_size,
        epochs=epochs,
        margin=margin,
        lambda_neg=lambda_neg
    )
    
    # return the trained model and final embeddings   
    return model, embeddings


if __name__ == "__main__":
    # generate some sample sensor data    
    num_samples = 100
    input_dim = 50
    num_classes = 5
    
    # create random feature vectors and labels    
    features = np.random.randn(num_samples, input_dim).astype(np.float32)
    labels = np.random.randint(0, num_classes, size=num_samples)
    
    # train the contrastive embedding model    
    print("Training contrastive embedding model...")
    model, embeddings = generate_sensor_embeddings_with_contrastive_learning(
        feature_vectors=features,
        labels=labels,
        embedding_dim=128, 
        epochs=20,              
        batch_size=16
    )
    
    print(f"Generated embeddings with shape: {embeddings.shape}")
    
    # verify embeddings have captured label structure using a simple check    
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.model_selection import train_test_split
    
    X_train, X_test, y_train, y_test = train_test_split(
        embeddings, labels, test_size=0.3, random_state=42
    )
    
    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(X_train, y_train)
    accuracy = knn.score(X_test, y_test)
    
    print(f"KNN classifier accuracy on embeddings: {accuracy:.4f}")