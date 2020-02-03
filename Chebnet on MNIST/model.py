import dgl.function as fn
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import time

from utils import chebyshev, rescale_L, set_device

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

    def forward(self, g, feature, L):
        V, featmaps = feature.size()
        Xt = torch.Tensor([]).to(self.device)

        for fmap in range(featmaps):
            Xt = torch.cat(
                (Xt, chebyshev(L, feature[:, fmap].view(-1, 1), self._k)), 0)

        g.ndata['h'] = Xt.view(V, -1)
        # print('DATA:',g.ndata['h'],g.ndata['h'].size())
        #g.update_all(msg, reduce)
        g.apply_nodes(func=self.apply_mod)
        return g.ndata.pop('h')

class Classifier(nn.Module):
    def __init__(self, in_dim, fc1,fc2, n_classifier, n_classes, k):
        super(Classifier, self).__init__()
        self.layers = nn.ModuleList([
            Chebyconv(in_dim, fc1, k),
            Chebyconv(fc1, fc2, k)])

        self.classify = nn.Sequential(
            nn.Linear(784, n_classifier),
            nn.ReLU(inplace=True),
            # nn.Dropout(), ## Add dropout later
            nn.Linear(n_classifier, n_classes)
        )

    def forward(self, g, L):
        B = g.batch_size
        h = g.ndata.pop('h').view(-1, 1)
        #for conv in self.layers:
            #h = conv(g, h, L)
        #g.ndata['h'] = h
        #hg = dgl.mean_nodes(g, 'h')
        #print('h',h,h.size())
        return self.classify(h.view(B, -1))