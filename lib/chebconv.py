import torch
import torch.nn as nn
import torch.nn.functional as F

import dgl
import dgl.function as fn


class NodeApplyModule(nn.Module):
    """Update the node feature hv with ReLU(Whv+b)."""

    def __init__(self, in_feats, out_feats, k, activation, bias=True):
        super(NodeApplyModule, self).__init__()
        self.linear = nn.Linear(k * in_feats, out_feats, bias)
        self.activation = activation

    def forward(self, node):
        h = self.linear(node.data['h'])
        if self.activation:
            h = self.activation(h)
        return {'h': h}


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
    activation : function, optional
        Activation function.
    bias : bool, optional
        If True, adds a learnable bias to the output. Default: ``True``.
    """

    def __init__(self,
                 in_feats,
                 out_feats,
                 k,
                 activation=F.relu,
                 bias=True):
        super(Cheb_Conv, self).__init__()
        self._k = k
        self._in_feats = in_feats
        self._out_feats = out_feats
        self.activation = activation
        self.apply_mod = NodeApplyModule(
            in_feats,
            out_feats,
            k=self._k,
            activation=self.activation)

    def forward(self, graph, signal, lambda_max=None):
        """
        Parameters
        ----------
        graph : DGLGraph
            The graph.
        feat : torch.Tensor [Nxf]
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

        def unnLaplacian(feat, D_sqrt, graph):
            """Operation D^-1/2 A D^-1/2, the DGL way."""
            graph.ndata['h'] = feat * D_sqrt
            graph.update_all(fn.copy_u('h', 'm'), fn.sum('m', 'h'))
            return graph.ndata.pop('h') * D_sqrt

        with graph.local_scope():
            D_sqrt = torch.pow(graph.in_degrees().float().clamp(
                min=1), -0.5).unsqueeze(-1).to(signal.device)

            if lambda_max is None:
                try:
                    lambda_max = dgl.laplacian_lambda_max(graph)
                except BaseException:
                    # if the largest eigonvalue is not found
                    lambda_max = torch.Tensor(2).to(signal.device)

            if isinstance(lambda_max, list):
                lambda_max = torch.Tensor(lambda_max).to(signal.device)
            if lambda_max.dim() == 1:
                lambda_max = lambda_max.unsqueeze(-1)  # (B,) to (B, 1)

            # broadcast from (B, 1) to (N, 1)
            lambda_max = dgl.broadcast_nodes(graph, lambda_max)

            # X_0(f)
            Xt = X_0 = signal

            # X_1(f)
            if self._k > 1:
                re_norm = 2. / lambda_max
                h = unnLaplacian(X_0, D_sqrt, graph)

                X_1 = - re_norm * h + X_0 * (re_norm - 1)
                Xt = torch.cat((Xt, X_1), 1)

            # Xi(x), i = 2...k
            for i in range(2, self._k):
                h = unnLaplacian(X_1, D_sqrt, graph)
                X_i = - 2 * re_norm * h + X_1 * 2 * (re_norm - 1) - X_0

                Xt = torch.cat((Xt, X_i), 1)
                X_1, X_0 = X_i, X_1

            # forward pass
            graph.ndata['h'] = Xt
            graph.apply_nodes(func=self.apply_mod)

            return graph.ndata.pop('h')
