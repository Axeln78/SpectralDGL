''' 
This file is a regroupment of functions that are supposed to create different lattices to test some hypothesis of laplacian proprieties

'''

# Libraries
import dgl
import networkx as nx

# --------- Lattices & regular tiling --------- #

# 2D regular lattice, connected to 4 neighbors
def regular_2D_lattice(size):
    g = dgl.DGLGraph()
    g.from_networkx(
        nx.grid_2d_graph(size, size))
    return g

# 2D regular lattice, connected to 8 neighbors 

def regular_2D_lattice_8_neighbors(size):
    g = dgl.DGLGraph()
    G = nx.grid_2d_graph(size, size)
    G.add_edges_from([
    ((x, y), (x+1, y+1))
    for x in range(size-1)
    for y in range(size-1)
    ] + [
    ((x+1, y), (x, y+1))
    for x in range(size-1)
    for y in range(size-1)
    ], weight=1)
    g.from_networkx(G)
    return g
    


# --------- Irregular tilings --------- #

def subgraph(g,parameter):
    
    return g

def contracted(g,contraction_list):
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
    
