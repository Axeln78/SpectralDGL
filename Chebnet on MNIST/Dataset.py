import dgl
import networkx as nx
import os
import torch


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
        self.graph = g

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


def check_mnist_dataset_exists(path_data='./'):
    flag_train_data = os.path.isfile(path_data + 'mnist/train_data.pt')
    flag_train_label = os.path.isfile(path_data + 'mnist/train_label.pt')
    flag_test_data = os.path.isfile(path_data + 'mnist/test_data.pt')
    flag_test_label = os.path.isfile(path_data + 'mnist/test_label.pt')
    if flag_train_data == False or flag_train_label == False or flag_test_data == False or flag_test_label == False:
        print('MNIST dataset missing - downloading...')
        import torchvision
        import torchvision.transforms as transforms
        trainset = torchvision.datasets.MNIST(root=path_data + 'mnist/temp', train=True,
                                              download=True, transform=transforms.ToTensor())
        testset = torchvision.datasets.MNIST(root=path_data + 'mnist/temp', train=False,
                                             download=True, transform=transforms.ToTensor())
        train_data = torch.Tensor(60000, 28, 28)
        train_label = torch.LongTensor(60000)

        for idx, example in enumerate(trainset):
            train_data[idx] = example[0].squeeze()
            train_label[idx] = example[1]

        torch.save(train_data, path_data + 'mnist/train_data.pt')
        torch.save(train_label, path_data + 'mnist/train_label.pt')
        test_data = torch.Tensor(10000, 28, 28)
        test_label = torch.LongTensor(10000)

        for idx, example in enumerate(testset):
            test_data[idx] = example[0].squeeze()
            test_label[idx] = example[1]

        torch.save(test_data, path_data + 'mnist/test_data.pt')
        torch.save(test_label, path_data + 'mnist/test_label.pt')
    return path_data


def datasampler(nb_selected_train_data, nb_selected_test_data):

    train_data = torch.load('mnist/train_data.pt').reshape(60000, 784)
    train_data = train_data[:nb_selected_train_data, :]
    #print(train_data.shape, type(train_data))

    train_labels = torch.load('mnist/train_label.pt')
    train_labels = train_labels[:nb_selected_train_data]
    # print(train_labels.shape)

    test_data = torch.load('mnist/test_data.pt').reshape(10000, 784)
    test_data = test_data[:nb_selected_test_data, :]
    # print(test_data.shape)

    test_labels = torch.load('mnist/test_label.pt')
    test_labels = test_labels[:nb_selected_test_data]
    # print(test_labels.shape)
    return train_data, train_labels, test_data, test_labels