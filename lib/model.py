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
    
    
    '''
    ------------------------------------------------------------------------------
    '''
    
class DGL_ChebConv(nn.Module):
    r"""Chebyshev Spectral Graph Convolution layer from paper `Convolutional
    Neural Networks on Graphs with Fast Localized Spectral Filtering
    <https://arxiv.org/pdf/1606.09375.pdf>`__.

    .. math::
        h_i^{l+1} &= \sum_{k=0}^{K-1} W^{k, l}z_i^{k, l}

        Z^{0, l} &= H^{l}

        Z^{1, l} &= \hat{L} \cdot H^{l}

        Z^{k, l} &= 2 \cdot \hat{L} \cdot Z^{k-1, l} - Z^{k-2, l}

        \hat{L} &= 2\left(I - \hat{D}^{-1/2} \hat{A} \hat{D}^{-1/2}\right)/\lambda_{max} - I

    Parameters
    ----------
    in_feats: int
        Number of input features.
    out_feats: int
        Number of output features.
    k : int
        Chebyshev filter size.
    bias : bool, optional
        If True, adds a learnable bias to the output. Default: ``True``.
    """

    def __init__(self,
                 in_feats,
                 out_feats,
                 k,
                 bias=True):
        super(DGL_ChebConv, self).__init__()
        self._in_feats = in_feats
        self._out_feats = out_feats
        self.fc = nn.ModuleList([
            nn.Linear(in_feats, out_feats, bias=False) for _ in range(k)
        ])
        self._k = k
        if bias:
            self.bias = nn.Parameter(th.Tensor(out_feats))
        else:
            self.register_buffer('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        """Reinitialize learnable parameters."""
        if self.bias is not None:
            init.zeros_(self.bias)
        for module in self.fc.modules():
            if isinstance(module, nn.Linear):
                init.xavier_normal_(module.weight, init.calculate_gain('relu'))
                if module.bias is not None:
                    init.zeros_(module.bias)

    def forward(self, graph, feat, lambda_max=None):
        '''
        
        ATTEMPT TO CORRECT THIS IMPLEMENTATION
        
        '''
        
        
        r"""Compute ChebNet layer.

        Parameters
        ----------
        graph : DGLGraph
            The graph.
        feat : torch.Tensor
            The input feature of shape :math:`(N, D_{in})` where :math:`D_{in}`
            is size of input feature, :math:`N` is the number of nodes.
        lambda_max : list or tensor or None, optional.
            A list(tensor) with length :math:`B`, stores the largest eigenvalue
            of the normalized laplacian of each individual graph in ``graph``,
            where :math:`B` is the batch size of the input graph. Default: None.
            If None, this method would compute the list by calling
            ``dgl.laplacian_lambda_max``.

        Returns
        -------
        torch.Tensor
            The output feature of shape :math:`(N, D_{out})` where :math:`D_{out}`
            is size of output feature.
        """
        with graph.local_scope():
            norm = th.pow(
                graph.in_degrees().float().clamp(min=1), -0.5).to(feat.device)
            if lambda_max is None:
                lambda_max = laplacian_lambda_max(graph)
            if isinstance(lambda_max, list):
                lambda_max = th.Tensor(lambda_max).to(feat.device)
            if lambda_max.dim() == 1:
                lambda_max = lambda_max  # (B,) to (B, 1)
            # broadcast from (B, 1) to (N, 1)
            lambda_max = broadcast_nodes(graph, lambda_max)
            # T0(X)
            Tx_0 = feat
            rst = self.fc[0](Tx_0)
            # T1(X)
            if self._k > 1:
                graph.ndata['h'] = Tx_0 * norm
                graph.update_all(fn.copy_u('h', 'm'), fn.sum('m', 'h'))
                h = graph.ndata.pop('h') * norm
                # Λ = 2 * (I - D ^ -1/2 A D ^ -1/2) / lambda_max - I
                #   = - 2(D ^ -1/2 A D ^ -1/2) / lambda_max + (2 / lambda_max - 1) I
                Tx_1 = -2. * h / lambda_max + Tx_0 * (2. / lambda_max - 1)
                rst = rst + self.fc[1](Tx_1)
            # Ti(x), i = 2...k
            for i in range(2, self._k):
                graph.ndata['h'] = Tx_1 * norm
                graph.update_all(fn.copy_u('h', 'm'), fn.sum('m', 'h'))
                h = graph.ndata.pop('h') * norm
                # Tx_k = 2 * Λ * Tx_(k-1) - Tx_(k-2)
                #      = - 4(D ^ -1/2 A D ^ -1/2) / lambda_max Tx_(k-1) +
                #        (4 / lambda_max - 2) Tx_(k-1) -
                #        Tx_(k-2)
                Tx_2 = -4. * h / lambda_max + Tx_1 * (4. / lambda_max - 2) - Tx_0
                rst = rst + self.fc[i](Tx_2)
                Tx_1, Tx_0 = Tx_2, Tx_1
            # add bias
            if self.bias is not None:
                rst = rst + self.bias
            return rst

