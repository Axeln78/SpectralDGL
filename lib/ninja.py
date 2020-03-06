'''
Quick functions for prototyping. All taking a dgl graph as argument and outputs some graphical represention
'''
import matplotlib.pyplot as plt
from laplacian import normalized_laplacian
import numpy.linalg
import networkx as nx
import numpy as np


def Gspyplot(Graph, Lambdamax=True):
    L = normalized_laplacian(Graph, Lambdamax)
    plt.imshow(L.to_dense(), cmap='viridis')
    plt.colorbar()
    return L


def eigondecomposition(g):
    G = g.to_networkx().to_undirected()
    L = nx.normalized_laplacian_matrix(G)
    e = numpy.linalg.eigvals(L.A)
    #e = nx.normalized_laplacian_spectrum(G)
    return e


def laplacianspecrum(L):
    e = numpy.linalg.eigvals(L)
    print("Largest eigenvalue:", max(e))
    print("Smallest eigenvalue:", min(e))
    plt.plot(np.sort(e), '.')  # histogram with 100 bins
    # plt.xlim(-0.2, 2.2)  # eigenvalues between 0 and 2
    plt.show()


def GeigplotH(Graph):
    e = eigondecomposition(Graph)
    print("Largest eigenvalue:", max(e))
    print("Smallest eigenvalue:", min(e))
    plt.hist(e, bins=150)  # histogram with 100 bins
    plt.xlim(-0.2, 2.2)  # eigenvalues between 0 and 2
    plt.show()


def GeigplotL(Graph):
    e = eigondecomposition(Graph)
    print("Largest eigenvalue:", max(e))
    print("Smallest eigenvalue:", min(e))
    plt.plot(np.sort(e), '.')  # histogram with 100 bins
    # plt.xlim(-0.2, 2.2)  # eigenvalues between 0 and 2
    plt.show()
