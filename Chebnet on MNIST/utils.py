import os
import sys
import torch
from scipy import sparse
#from dgl.batched_graph import BatchedDGLGraph, unbatch
import dgl
import torch.sparse
import numpy as np


def save_model(name, model):
    if not os.path.exists('saved_models'):
        os.makedirs('saved_models')
    torch.save(model.state_dict(), './saved_models/' + name + '.pt')


def chebyshev(L, X, K):
    """Return T_k X where T_k are the Chebyshev polynomials of order up to K.
    Complexity is O(KMN).
    -  - - - - - - - - - 
    - Here implemented for torch
    - L should be on device!
    """
    M, N = X.shape
    #assert L.dtype == X.dtype

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