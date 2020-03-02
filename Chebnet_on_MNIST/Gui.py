import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.widgets import Slider
import networkx as nx
import numpy
#%matplotlib qt

from graphs import regular_2D_lattice, regular_2D_lattice_8_neighbors, random_edge_suppression

g = regular_2D_lattice(5)

g2 = regular_2D_lattice_8_neighbors(5)

g3 = random_edge_suppression(5,5)


G2 = g2.to_networkx().to_undirected()
L2 = nx.normalized_laplacian_matrix(G2)
e = numpy.linalg.eigvals(L2.A)

fig, ax = plt.subplots()

s = np.sort(e)
l, = plt.plot(np.sort(nx.normalized_laplacian_spectrum(g.to_networkx().to_undirected()))) 
l, = plt.plot(np.sort(e), lw=2)

ax = plt.axis([0,len(e),0,np.max(e)*1.5])

axamp = plt.axes([0.25, .03, 0.50, 0.02])
# Slider
samp = Slider(axamp, 'Theta', 0, 1.5, valinit=1)

def update(val):
    # amp is the current value of the slider
    amp = samp.val
    # update curve
    l.set_ydata(np.sort(numpy.linalg.eigvals((L2*amp).A)))
    # redraw canvas while idle
    fig.canvas.draw_idle()

# call update function on slider value change
samp.on_changed(update)

plt.show()