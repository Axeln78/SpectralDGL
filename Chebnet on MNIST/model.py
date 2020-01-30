import dgl.function as fn
import torch
import torch.nn as nn
import torch.nn.functional as F
#import networkx as nx
import dgl
import time 

from utils import chebyshev,rescale_L

def set_device():
    if torch.cuda.is_available():
        print('cuda available')
        dtypeFloat = torch.cuda.FloatTensor
        dtypeLong = torch.cuda.LongTensor
        torch.cuda.manual_seed(1)
        device=torch.device('cuda:0')
    else:
        print('cuda not available')
        dtypeFloat = torch.FloatTensor
        dtypeLong = torch.LongTensor
        torch.manual_seed(1)
        device=torch.device('cpu')
    return device
    
# Sends a message of node feature h.
msg = fn.copy_src(src='h', out='m')

def reduce(nodes):
    """Take a sum over all neighbor node features hu and use it to
    overwrite the original node feature."""
    accum = torch.sum(nodes.mailbox['m'], 1)
    return {'h': accum}


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
        # Parameters
        self._in_feats = in_feats
        self._out_feats = out_feats
        self._k = k 
        # Convlayer
        self.apply_mod = NodeApplyModule(
            in_feats*k, out_feats, activation=F.relu)
        self.device = set_device()

        ##
        # NEED TO MAKE A PARAMETER INITIALIZER /!\
        ##

    def forward(self, g, feature, L):
        V, featmaps = feature.size()
        Xt = torch.Tensor([]).to(self.device)
        
        for fmap in range(featmaps):
            #print('Xt',Xt,'L',self.L,'fmap',feature[:, fmap].view(-1, 1))
            Xt = torch.cat((Xt, chebyshev( L, feature[:, fmap].view(-1, 1), self._k)), 0)
            
        g.ndata['h'] = Xt.view(V, -1)
        g.update_all(msg, reduce)
        g.apply_nodes(func=self.apply_mod)
        return g.ndata.pop('h')
    

# NEED TO CODE A WEIGHT INITIALIZER

class Classifier(nn.Module):
    def __init__(self, in_dim, hidden_dim, n_classes, k):
        super(Classifier, self).__init__()
        self.layers = nn.ModuleList([
            # HARD CODE THIS FOR THE MOMENT
            Chebyconv(in_dim, 32, k),
            Chebyconv(32, hidden_dim, k)])

        self.classify = nn.Sequential(
            nn.Linear(hidden_dim, 50),
            nn.ReLU(inplace=True),
            #nn.Dropout(), ## Add dropout when there will be enough features
            nn.Linear(50, n_classes)
        ) 

    def forward(self, g, L):
        h = g.ndata.pop('h').view(-1, 1)
        t = time.time() 
        for conv in self.layers:
            h = conv(g, h, L) 
        g.ndata['h'] = h
        hg = dgl.mean_nodes(g, 'h') 
        return self.classify(hg)