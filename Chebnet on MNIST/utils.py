import os
import sys
import torch
from scipy import sparse
from dgl.batched_graph import BatchedDGLGraph, unbatch
import dgl
import torch.sparse
import numpy as np


def save_model(name, model):
    if not os.path.exists('saved_models'):
        os.makedirs('saved_models')
    torch.save(model.state_dict(), './saved_models/' + name + '.pt')


def rescale_L(L, lmax=2):
    """
    Rescale the Laplacian eigenvalues in [-1,1]. 
    here implemented for torch
    """
    M, M = L.shape
    I = sparse.eye(M)
    L = L / (lmax / 2)
    #L = L - I
    return L


def chebyshev(L, X, K):
    """Return T_k X where T_k are the Chebyshev polynomials of order up to K.
    Complexity is O(KMN).
    -  - - - - - - - - - 
    - Here implemented for torch
    - Change to sparse torch?
    - Rescaled L needed
    - try no grad?
    """
    M, N = X.shape
    #assert L.dtype == X.dtype
    '''

    Xt = torch.empty((K, M, N), dtype=L.dtype)

    Xt[0, :] = X
    if K > 1:
        Xt[1, ...] = torch.mm(L, X)t
        #Xt[1, ...] = my_sparse_mm.apply(L, X)
        for k in range(2, K):
            var = 2 * torch.mm(L, Xt[k-1, :]) - Xt[k-2, :]
            Xt[k, :] = var
    return Xt

    '''
    #L = L.to(device)
    x0 = X
    x = x0.unsqueeze(0)
    if K > 1:
        # print('L',L,'x0',x0)
        x1 = torch.sparse.mm(L, x0)              # V x Fin*B
        x = torch.cat((x, x1.unsqueeze(0)), 0)  # 2 x V x Fin*B
    for k in range(2, K):
        x2 = 2 * torch.sparse.mm(L, x1) - x0
        x = torch.cat((x, x2.unsqueeze(0)), 0)  # M x Fin*B
        x0, x1 = x1, x2

    return x


def laplacian(g):
    '''
    Function that computes the normalized laplacian anf the largest eigonvalues of a graph g 

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
        laplacian = - norm * adj * norm  # sparse.eye(n) -
        L.append(laplacian)
        rst.append(sparse.linalg.eigs(laplacian, 1, which='LM',
                                      return_eigenvectors=False)[0].real)

    L_out = sparse.block_diag(L)
    return npsparse_to_torch(L_out), rst


def npsparse_to_torch(tensor):
    '''
    Convert sparce matrix to tensor format. There is no torch built in function for that.

    Parameters
    ----------
    Tensor : type np.sparse

    Return
    ----------
    Tensor : type Torch.sparse.FloatTensor

    '''
    coo = tensor.tocoo()

    values = coo.data
    indices = np.vstack((coo.row, coo.col))

    i = torch.LongTensor(indices)
    v = torch.FloatTensor(values)
    shape = coo.shape

    res = torch.sparse.FloatTensor(i, v, torch.Size(shape))
    return res


def set_device(verbose=False):
    if torch.cuda.is_available():
        if verbose:
            print('cuda available')
        dtypeFloat = torch.cuda.FloatTensor
        dtypeLong = torch.cuda.LongTensor
        torch.cuda.manual_seed(1)
        device = torch.device('cuda:0')
    else:
        if verbose:
            print('cuda not available')
        dtypeFloat = torch.FloatTensor
        dtypeLong = torch.LongTensor
        torch.manual_seed(1)
        device = torch.device('cpu')
    return device