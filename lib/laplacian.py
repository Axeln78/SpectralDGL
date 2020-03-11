import dgl
from dgl.batched_graph import BatchedDGLGraph, unbatch

#import torch.sparse
from scipy import sparse

from utils import npsparse_to_torch
'''
All laplacian operations
'''


def normalized_laplacian(g, decomp=True):
    '''
    Function that computes the normalized laplacian (sparse) and the largest eigonvalues of a graph g

    Parameters
    ----------
    G : graph
       A DGL graph Batched or not

    Returns
    -------
    L: NumPy sparse matrix
      The normalized Laplacian matrix of G.

    lmax: Array of largest eigonvalues

    '''

    if isinstance(g, BatchedDGLGraph):
        g_arr = unbatch(g)
    else:
        g_arr = [g]

    rst = []
    L = []
    for g_i in g_arr:
        n = g_i.number_of_nodes()
        adj = g_i.adjacency_matrix_scipy(return_edge_ids=False).astype(float)
        norm = sparse.diags(dgl.backend.asnumpy(
            g_i.in_degrees()).clip(1) ** -0.5, dtype=float)
        laplacian = sparse.eye(n) - norm * adj * norm
        if decomp:
            lambda_max = sparse.linalg.eigs(laplacian, 1, which='LM',
                                            return_eigenvectors=False)[0].real
            rst.append(lambda_max)
            laplacian = rescale_L(laplacian, lambda_max)
        else:
            laplacian = rescale_L(laplacian)
        L.append(laplacian)

    L_out = sparse.block_diag(L)
    # rst is the vector with all highest eigvals
    # It can also be an output
    return npsparse_to_torch(L_out)#, rst


def rescale_L(L, lmax=2):
    """
    Rescale the Laplacian eigenvalues in [-1,1].
    here implemented for torch
    """
    M, M = L.shape
    I = sparse.eye(M)
    L = L / (lmax / 2)
    L = L - I
    return L
