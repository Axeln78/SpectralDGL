import torch
import torch.nn as nn
import torch.nn.functional as F

import dgl

from layers.gcn_layer import GCNLayer
from layers.mlp_readout_layer import MLPReadout

class ChebNet(nn.Module):
    def __init__(self, net_params):
        super().__init__()
        in_dim = net_params['in_dim']
        hidden_dim = net_params['hidden_dim']
        out_dim = net_params['out_dim']
        n_classes = net_params['n_classes']
        in_feat_dropout = net_params['in_feat_dropout']
        dropout = net_params['dropout']
        n_layers = net_params['L']
        self.readout = net_params['readout']
        self.graph_norm = net_params['graph_norm']
        self.batch_norm = net_params['batch_norm']
        self.residual = net_params['residual']
        
        self.embedding_h = nn.Linear(in_dim, hidden_dim)
        self.in_feat_dropout = nn.Dropout(in_feat_dropout)
        
        self.layers = nn.ModuleList([GCNLayer(hidden_dim, hidden_dim, F.relu, dropout,
                                              self.graph_norm, self.batch_norm, self.residual) for _ in range(n_layers-1)])
        self.layers.append(GCNLayer(hidden_dim, out_dim, F.relu, dropout, self.graph_norm, self.batch_norm, self.residual))
        
        self.MLP = nn.Sequential(
            nn.Linear(out_dim, n_classifier), 
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(n_classifier, n_classes)
        )       

    def forward(self, g, h, e, snorm_n, snorm_e):
        # Do something with the signal here
        h = self.embedding_h(h)
        h = self.in_feat_dropout(h)
        
        # Put it through the conv layers
        for conv in self.layers:
            h = conv(g, h, snorm_n)
        g.ndata['h'] = h
        
        # Global pooling
        if self.readout == "sum":
            hg = dgl.sum_nodes(g, 'h')
        elif self.readout == "max":
            hg = dgl.max_nodes(g, 'h')
        elif self.readout == "mean":
            hg = dgl.mean_nodes(g, 'h')
        else:
            hg = dgl.mean_nodes(g, 'h')  # default readout is mean nodes
            
        # Send through MLP
        return self.MLP(hg)
    
    def loss(self, pred, label):
        criterion = nn.CrossEntropyLoss()
        loss = criterion(pred, label)
        return loss