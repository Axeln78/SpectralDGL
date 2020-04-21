import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F

from chebconv import Cheb_Conv

class Cheb_ZINC(nn.Module):

    def __init__(self, num_atom_type, fc2, n_classifier, n_classes, k, readout="sum"):
        super(Cheb_ZINC, self).__init__()
        '''
        Parameters:
        -----------
        in_dim: Number of input features
        fc1: number of output from the first convolutional layer
        fc2: number of outputs from the second conv layer
        n_classifier number of hidden layers in the classifier module
        n_classes: number of classes for the classifier
        k: number of Chebynomes to compute

        '''
        self.readout = readout
        
        self.layers = nn.ModuleList([
            Cheb_Conv(fc2, fc2, k),
            Cheb_Conv(fc2, 2*fc2, k),
            Cheb_Conv(2*fc2, 4*fc2, k)])
            
        self.MLP = nn.Sequential(
            nn.Linear(4*fc2, 4*fc2),
            nn.ReLU(inplace=True),
            #nn.Dropout(p=0.5),
            nn.Linear(4*fc2, n_classes)
        )
        
        self.embedding_h = nn.Embedding(num_atom_type, fc2)

    def forward(self, g, signal, lambda_max=None):
        '''
        Parameters:
        -----------
        g: graph
        signal : signal over graph 
        '''
        

        batch_size = g.batch_size # if isinstance(g, BatchedDGLGraph) else 1
        h = signal # [B*V*h,1] = [V2*h,1]
        
        h = self.embedding_h(h.long()).squeeze()
        
        for conv in self.layers:
            h = conv(g, h, lambda_max)
        g.ndata['h'] = h
        
        if self.readout == "sum":
            hg = dgl.sum_nodes(g, 'h')
        elif self.readout == "max":
            hg = dgl.max_nodes(g, 'h')
        elif self.readout == "mean":
            hg = dgl.mean_nodes(g, 'h')
        else:
            hg = dgl.mean_nodes(g, 'h') 
        
        return self.MLP(hg)  # [B x Fc2]