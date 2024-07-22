import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class lca_alignment_loss(nn.Module):
    def __init__(self, tree,lca_matrix,alignment_mode=2):
        super().__init__()
        self.lca_matrix=lca_matrix
        self.tree=tree
        self.alignment_mode=alignment_mode
        self.reverse_matrix=1-self.lca_matrix
    def forward(self,logits,targets,lambda_weight=0.03):
        reverse_matrix=self.reverse_matrix

        # Ensure distance matrix is float32
        distance_matrix = self.lca_matrix.float()
    
        # Compute the predicted probabilities
        probs = F.softmax(logits, dim=1)
        # One-hot encode the targets
        one_hot_targets = F.one_hot(targets, num_classes=logits.size(1)).float()
        
        # Compute the standard cross-entropy loss
        standard_loss = -torch.sum(one_hot_targets * torch.log(probs + 1e-9), dim=1)
        

        # Compute the alignment soft loss
        if self.alignment_mode==0: # not using alignment soft loss
            total_loss=standard_loss
        else:
            if self.alignment_mode==1: # BCE-form alignment soft loss
                criterion=nn.BCEWithLogitsLoss(reduction='none')
                alignment_loss = torch.mean(criterion(logits, reverse_matrix[targets]),dim=1)
            elif self.alignment_mode==2: # CE-form alignment soft loss
                alignment_loss=-torch.mean(reverse_matrix[targets] * torch.log(probs + 1e-9), dim=1)
            assert lambda_weight<=1
            assert lambda_weight>=0
            total_loss = lambda_weight* standard_loss + alignment_loss
        
        # Return the mean loss over the batch
        return torch.mean(total_loss)



