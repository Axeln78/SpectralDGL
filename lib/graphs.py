'''
This file is a regroupment of functions that are supposed to create different lattices to test some hypothesis of laplacian proprieties

'''

# Libraries
import dgl
import networkx as nx
import random
import copy
import dgl
#from dgl import BatchedDGLGraph, unbatch
import torch


def transform(g):
    '''
    takes in a nx graph and returns a dgl Graph
    '''
    G = dgl.DGLGraph()
    G.from_networkx(g)
    return G

# --------- Lattices & regular tiling --------- #

# 2D regular lattice, connected to 4 neighbors


def regular_2D_lattice(size):
    g = nx.grid_2d_graph(size, size)
    return transform(g)


def regular_2D_lattice_nx(size):
    g = nx.grid_2d_graph(size, size)
    return g

# 2D regular lattice, connected to 8 neighbors


def regular_2D_lattice_8_neighbors(size):
    G = nx.grid_2d_graph(size, size)
    G.add_edges_from([
        ((x, y), (x + 1, y + 1))
        for x in range(size - 1)
        for y in range(size - 1)
    ] + [
        ((x + 1, y), (x, y + 1))
        for x in range(size - 1)
        for y in range(size - 1)
    ], weight=1)
    return transform(G)


# --------- Irregular tilings --------- #

def random_edge_suppression(size, k):
    '''
    Takes a DGLGraph and K a number of edges to remove
    '''
    g = nx.grid_2d_graph(size, size)
    to_remove = random.sample(g.edges(), k)
    g.remove_edges_from(to_remove)
    return transform(g)


def random_edge_suppression_nx(G, k):
    '''
    Takes a NX and K a number of edges to remove
    '''

    to_remove = random.sample(G.edges(), k)
    G.remove_edges_from(to_remove)

    return transform(G)


def transform_default(g, func, arg):
    '''
    Takes a DGLGraph and K a number of edges to remove
    '''
    if isinstance(g, BatchedDGLGraph):
        g_arr = dgl.unbatch(g)
    else:
        g_arr = [g]

    G = []
    for g_i in g_arr:
        g_i = func(g_i, arg)
        G.append(g_i)
    return dgl.batch(G)


def random_node_suppression(g, k):

    if isinstance(g, BatchedDGLGraph):
        g_arr = dgl.unbatch(g)
    else:
        g_arr = [g]

    G = []
    for g_i in g_arr:
        n_nodes = len(g_i)
        l = [random.randint(0, n_nodes) for i in range(k)]
        g_i.remove_nodes(l)
        diff = n_nodes - len(g_i)
        g_i.add_nodes(diff, {'h': torch.zeros(diff)})
        G.append(g_i)
    return dgl.batch(G)

    return g


'''
def random_node_suppression(g, k):
    n_nodes = len(g)
    l = [random.randint(0, n_nodes) for i in range(k)]
    g.remove_nodes(l)
    diff = n_nodes - len(g)
    g.add_nodes(diff, {'h': torch.zeros(diff)})
    return g
'''


def random_geometric_graph(size, p=0.058):
    '''
    size: sqrt of number of nodes
    p: max distance between two nodes to be connected
    '''
    g = nx.random_geometric_graph(size * size, p)

    return transform(g)


def random_waxman_graph(size):
    '''
    size: sqrt of number of nodes
    p: max distance between two nodes to be connected
    '''
    g = nx.waxman_graph(size * size, alpha=1, beta=0.015)

    return transform(g)


def contracted(g, contraction_list):
    '''https://networkx.github.io/documentation/networkx-1.10/reference/generated/networkx.algorithms.minors.contracted_nodes.html#networkx.algorithms.minors.contracted_nodes

    Parameters:
    -----------
    G : graph of type DGLGraph

    contraction_list : list of node to contract. A list element (1,2) will merge node 1 and 2.
    '''
    outG = dgl.DGLGraph()
    nx_g = g.to_networkx().to_undirected()

    for elem in contraction_list:
        h = nx.contracted_edge(nx_g, elem)

    outG.from_networkx(h)
    return outG
