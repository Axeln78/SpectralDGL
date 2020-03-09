# Spectral DGL

## Lib
Regroupment of functions used in all of the project
```bash
lib/
├── dataset.py
├── graphs.py
├── laplacian.py
├── model.py
├── ninja.py
└── utils.py
```

## Resources

Shared resources between different part of the project

- imgs folder are for exporting figures that can be included in the final report 

- models regroup several trained ChebNets that are all supposedly achieving 98% accuracy over their testing set

```bash
├── imgs
├── models
```

## Chebnet on MNIST
Notebook implementation of the Chebyconv using the DGL framework. 
The net is a simple LeNet 5 with three convolution layer and two linear layers.
The dataset is the good-old MNIST
```bash
├──ChebGCNs.ipynb
├── README.md
├── Eigonvalues_Viz.ipynb
├── Gui.py
├── Training_Chebnet_on_MNIST.ipynb
├── filtervisualisation.ipynb
├── laplacianprediction.ipynb
├── transform_training.ipynb
└── visualisations.ipynb
```

## Stochastic learning

Notebook for stochastic learning. For now only two notebooks to train and test the model are implemented.

```bash
├── Training
├── Testing
```

