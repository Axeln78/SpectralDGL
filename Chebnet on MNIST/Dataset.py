import dgl
import networkx as nx

class MNISTDataset(object):
    """The dataset class.

    .. note::
        This dataset class is compatible with pytorch's :class:`Dataset` class.

    Parameters
    ----------
    data: Torch tensor
        Signal over the graph, here shades of grey
    labels: int
        handwritten digit
    lattice_size: int
        number of pixel on one dimention of the original image
    """

    def __init__(self, data, labels, lattice_size=28):
        super(MNISTDataset, self).__init__()
        self.data = data
        self.labels = labels

        # Define the regular lattice graph used for ALL computation
        self.graph = []
        g = dgl.DGLGraph()
        g.from_networkx(
            nx.grid_2d_graph(lattice_size, lattice_size))
        self.graph= g

    def __getitem__(self, idx):
        """Get the i^th sample.

        Paramters
        ---------
        idx : int
            The sample index.

        Returns
        -------
        (dgl.DGLGraph, int)
            The graph with the signal on "['h']" channel and its label.
        """
        oG = self.graph
        oG.ndata['h'] = self.data[idx]

        return oG, self.labels[idx]

    @property
    def num_classes(self):
        """Number of classes from 0 to 9."""
        return 10

    def __len__(self):
        """Return the number of graphs in the dataset."""
        return len(self.labels)
