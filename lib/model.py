import dgl.function as fn
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import time

from utils import chebyshev, set_device
from dgl.batched_graph import BatchedDGLGraph

'''
CODE not used for the moment
--------

# Sends a message of node feature h.
msg = fn.copy_src(src='h', out='m')

def reduce(nodes):
    """Take a sum over all neighbor node features hu and use it to
    overwrite the original node feature."""
    accum = torch.sum(nodes.mailbox['m'], 1)
    return {'h': accum}
    
'''


class NodeApplyModule(nn.Module):
    """Update the node feature hv with ReLU(Whv+b)."""

    def __init__(self, in_feats, out_feats, activation, bias=True):
        super(NodeApplyModule, self).__init__()
        self.linear = nn.Linear(in_feats, out_feats, bias)
        self.activation = activation

    def forward(self, node):
        h = self.linear(node.data['h'])
        h = self.activation(h)
        return {'h': h}


class Chebyconv(nn.Module):
    def __init__(self, in_feats, out_feats, k, bias=True):
        super(Chebyconv, self).__init__()
        '''
        Parameters:
        -----------
        in_feats: number of input features
        out_feats: number of outpout features
        k: order of chebynome
        '''
        
        # Parameters
        self._in_feats = in_feats
        self._out_feats = out_feats
        self._k = k
        # Convlayer
        self.apply_mod = NodeApplyModule(
            in_feats*k, out_feats, activation=F.relu)
        self.device = set_device()

    def forward(self, g, feature, L): 
        V, featmaps = feature.size()
        Xt = torch.Tensor([]).to(self.device)
        
        for fmap in range(featmaps):
            Xt = torch.cat(
                (Xt, chebyshev(L, feature[:, fmap].view(-1, 1), self._k)), 0) 
            
        g.ndata['h'] = Xt.squeeze().t() 
        g.apply_nodes(func=self.apply_mod)
        return g.ndata.pop('h')


class Classifier(nn.Module):
    
    def __init__(self, in_dim, fc1, fc2, n_classifier, n_classes, k):
        super(Classifier, self).__init__()
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

        self.layers = nn.ModuleList([
            Chebyconv(in_dim, fc1, k),
            Chebyconv(fc1, fc2, k)])

        self.classify = nn.Sequential(
            nn.Linear(fc2*784, n_classifier),
            nn.ReLU(inplace=True),
            nn.Dropout(), ## Add dropout later
            nn.Linear(n_classifier, n_classes)
        )

    def forward(self, g, signal, L):
        '''
        Parameters:
        -----------
        g: graph
        signal : signal over graph
        L : Laplacian matrix
        '''
        
        batch_size = g.batch_size if isinstance(g, BatchedDGLGraph) else 1
        h = signal.view(-1,1)  #[B*V*h,1] = [V2*h,1]
        for conv in self.layers:
            h = conv(g, h, L)
        #g.ndata['h'] = h
        #hg = dgl.sum_nodes(g, 'h') # SUM OVER NODES?
        return self.classify(h.view(batch_size, -1)) # {B, V*F} correctly?

class SmallCheb(nn.Module):
    
    def __init__(self, in_dim, fc1, n_classifier, n_classes, k):
        super(SmallCheb, self).__init__()
        '''
        Parameters:
        -----------
        in_dim: Number of input features
        fc1: number of output from the first convolutional layer
        n_classifier number of hidden layers in the classifier module
        n_classes: number of classes for the classifier
        k: number of Chebynomes to compute

        '''

        self.layers = nn.ModuleList([
            Chebyconv(in_dim, fc1, k)])

        self.classify = nn.Sequential(
            nn.Linear(fc1*784, n_classifier),
            nn.ReLU(inplace=True),
            nn.Dropout(), ## Add dropout later
            nn.Linear(n_classifier, n_classes)
        )

    def forward(self, g, signal, L):
        '''
        Parameters:
        -----------
        g: graph
        signal : signal over graph
        L : Laplacian matrix
        '''
        
        batch_size = g.batch_size if isinstance(g, BatchedDGLGraph) else 1
        h = signal.view(-1,1)  #[B*V*h,1] = [V2*h,1]
        for conv in self.layers:
            h = conv(g, h, L)
        return self.classify(h.view(batch_size, -1)) # {B, V*F} correctly?


    
class SanityCheck(nn.Module):
    def __init__(self, in_dim, n_classifier, n_classes):
        super(TestModule, self).__init__()
        self.fc1 = nn.Linear(in_dim, in_dim)
        self.fc2 = nn.Linear(in_dim, n_classifier)
        self.fc3 = nn.Linear(n_classifier, n_classes)
        
    def forward(self, g, s, L):
        batch_size = g.batch_size if isinstance(g, BatchedDGLGraph) else 1
        x = F.relu(self.fc1(s.view(batch_size,-1)))
        x = F.relu(self.fc2(x)) 
        x = self.fc3(x) 
        return x
