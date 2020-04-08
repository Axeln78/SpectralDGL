import os
import sys
import torch
from scipy import sparse
import dgl
import torch.sparse
import numpy as np


def save_model(name, model):
    if not os.path.exists('../resources/models/'):
        os.makedirs('../resources/models/')
    torch.save(model.state_dict(), '../resources/models/' + name + '.pt')


def load_model(name):
    assert os.path.exists('../resources/models/'), "Directory not found!"
    return torch.load(
        "../resources/models/" + name + ".pt",
        map_location=set_device())


def collate(samples):
    '''
    Function that helps with the overall collation of graph, signals and labels

    Return
    ----------
    Batched graph, labels (torch.tensor), signals (torch.tensor)
    
    '''
    # The input `samples` is a list of pairs
    #  (graph, label, signal).
    graphs, labels, signals = map(list, zip(*samples))
    batched_graph = dgl.batch(graphs)
    return batched_graph, torch.tensor(labels), torch.stack(signals).view(-1)

def collate2(samples):
    '''
    Function that helps with the overall collation of graph, signals and labels

    Return
    ----------
    Batched graph, labels (torch.tensor), signals (torch.tensor)
    
    '''
    # The input `samples` is a list of pairs
    #  (graph, label, signal).
    graphs, labels = map(list, zip(*samples))
    batched_graph = dgl.batch(graphs)
    return batched_graph, torch.tensor(labels)


def chebyshev(L, X, K):
    """
    Return T_k X where T_k are the Chebyshev polynomials of order up to K.
    Complexity is O(KMN).
    - --------------------------
    - Here implemented for torch
    - L should be on device!
    """
    M, N = X.shape
    #assert L.dtype == X.dtype

    x0 = X
    x = x0.unsqueeze(0)
    if K > 1:
        x1 = torch.mm(L, x0)              # V x Fin*B
        x = torch.cat((x, x1.unsqueeze(0)), 0)  # 2 x V x Fin*B
    for k in range(2, K):
        x2 = 2 * torch.mm(L, x1) - x0
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
