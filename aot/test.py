import torch 
import sys
def forward(self, embeddings, labels, labelWeights=[1.0,1.0,1.0]):

    batch_size = embeddings.size(0)
    
    # compute pairwise distances        
    dist_matrix = torch.cdist(embeddings, embeddings, p=2)
    
    # create mask for positive pairs (same label)    Note that there are multiple labels so the matrices need to be shifted!
    #Multiple labels introduces a 2D matrix instead of a 1d vector, which means a 3D matrix is resultant instead of a 2d matrix.
    #this means each Z matrix is a mask for a given label and their diagonal needs to be 0'd
    #this also means each label row needs to ONLY be compared to itself and not other rows.
    #note this needs to be modified so that multi-dimensional labels are transposed instead of unsqueezed
    #then for the similarity, we will add each matrix together
    #NOTICE!!!!  Because the columns are NOT INDEPENDANT, a weighting for each label column is necessary (defaulted to 1 for each)!!
    print(f'label shape!!:{labels.shape}\n') #batch_size x # labels in torch its row x col
    print(labels[0])
    if labels.shape==1:
        pos_mask = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()#this fails because you need to ONLY compare the row of a matrix with the corresponding columnof the transpose (i.e. each label's sample is compared against each other and a mask is generated)
    else:
        pos_mask=[]
        for i in range(labels.size(1)):
            pos_mask_mat=(labels[:,0] ==torch.transpose(labels[:,0],0,1)).float()*labelWeights[i]
            #pos_mask=(labels == torch.transpose(labels,0,1)).float()
            pos_mask.append(pos_mask_mat)        
    for i in range(pos_mask.size(0)):
        torch.Tensor.fill_diagonal_(pos_mask[i], 0)# exclude self-pairs 
        #pos_mask.fill_diagonal_(0)  # exclude self-pairs        
    
    # create mask for negative pairs (different label)       
    if labels.shape==1: 
        neg_mask = (labels.unsqueeze(0) != labels.unsqueeze(1)).float()
    else:
        neg_mask=[]
        for i in range(labels.size(1)):
            neg_mask_mat=(labels[:,0] != torch.transpose(labels[:,0],0,1)).float()*labelWeights[i]
        #neg_mask = (labels != torch.transpose(labels,0,1)).float()
    
    sys.exit('checking matrix sizes')
    #loss computation: note this sums up over ALL LABELS
    # compute positive pair loss: pull similar patterns together        
    pos_loss = (dist_matrix * pos_mask).sum() / (pos_mask.sum() + 1e-8)
    
    # compute negative pair loss: push dissimilar patterns apart        
    neg_loss = torch.clamp(self.margin - dist_matrix, min=0) * neg_mask
    neg_loss = neg_loss.sum() / (neg_mask.sum() + 1e-8)
    
    # combined loss        
    loss = pos_loss + self.lambda_neg * neg_loss
    
    return loss