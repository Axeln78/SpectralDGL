import torch
import torch.nn as nn
import torch.nn.functional as F

import dgl

'''
class NodeApplyModule(nn.Module):
    """Update the node feature hv with ReLU(Whv+b)."""

    def __init__(self, in_feats, out_feats, activation, bias=True):
        super(NodeApplyModule, self).__init__()
        self.linear = nn.Linear(in_feats, out_feats, bias)
        #self.activation = activation

    def forward(self, node):
        h = self.linear(node.data['h'])
        #h = self.activation(h)
        return {'h': h}
'''

class Cheb_Conv(nn.Module):
    """
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
        super(ChebConv, self).__init__()
        self._k = k
        self._in_feats = in_feats
        self._out_feats = out_feats
        self.fc = nn.Linear(in_feats*k, out_feats, bias)
        #self.apply_mod = NodeApplyModule(
        #    in_feats*k, out_feats, activation=F.relu)

    def forward(self, graph, signal, lambda_max=None):
        
        def lin_layer(nodes): return {'h': self.fc(node.data['h'])}
   
        Xt = torch.Tensor([]) #.to(self.device)
 
        with graph.local_scope():
            # Calc D
            D_sqrt = th.pow(
                graph.in_degrees().float().clamp(min=1), -0.5).unsqueeze(-1).to(signal.device)
             
            if lambda_max is None:
                lambda_max = dgl.laplacian_lambda_max(graph) #try catch here?
            if isinstance(lambda_max, list):
                lambda_max = th.Tensor(lambda_max).to(signal.device)
            if lambda_max.dim() == 1:
                lambda_max = lambda_max.unsqueeze(-1)  # (B,) to (B, 1)
            
            # broadcast from (B, 1) to (N, 1)
            lambda_max = broadcast_nodes(graph, lambda_max)
        
            # X_0(f)
            X_0 = signal 
            Xt.append(X_0)
            
            def unnLaplacian(signal,D_sqrt,graph):
                graph.ndata['h'] = signal * D_sqrt
                graph.update_all(fn.copy_u('h', 'm'), fn.sum('m', 'h'))
                return graph.ndata.pop('h') * D_sqrt
            
            # X_1(f)
            if self._k > 1:
                re_norm = 2./ lambda_max
                
                h = unnLaplacian(X_0,D_sqrt,graph)
                '''
                graph.ndata['h'] = X_0 * D_sqrt
                graph.update_all(fn.copy_u('h', 'm'), fn.sum('m', 'h'))
                h = graph.ndata.pop('h') * D_sqrt
                '''
                X_1 = - re_norm * h + X_0 * (re_norm - 1) 
                Xt.append(X_1)
                
            # Xi(x), i = 2...k
            for i in range(2, self._k):
                
                h = unnLaplacian(X_1,D_sqrt,graph)
                '''
                graph.ndata['h'] = X_1 * D_sqrt
                graph.update_all(fn.copy_u('h', 'm'), fn.sum('m', 'h'))
                h = graph.ndata.pop('h') * D_sqrt
                 '''
                X_i = - 2 * re_norm * h  + X_1 * 2 * (re_norm - 1) - X_0
                
                Xt.append(X_i)
                X_1, X_0 = X_i, X_1
            
            # forward pass
            g.ndata['h'] = Xt.squeeze().t() 
            
            g.apply_nodes(func=lin_layer)
            # or #g.ndata['h'] = self.linear(g.ndata['h'])
            
            return g.ndata.pop('h')
        
        