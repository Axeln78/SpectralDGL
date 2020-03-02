# Transferability of spectral Graph Neural Networks 

This private repository regroups the efforts to be able to study the transferability of graph neural networks by learning a laplacian transformation

## Python Files:

* Dataset.py - torch dataset definition
* model.py - torch implementation of the Chebnet
* utils.py - utility functions
* laplacian.py - Group all functions linked with laplacian calculation
* graphs.py - Group all graph generation functions

## Support Notebooks
Notebooks made to visualise, measure of develop different scripts.

* ChebGCNs - Chebnet training notebook

* visualisation - Look at the Chebyshev polynomials of a given graph / signal.
![alt text]('Chebfilters.png')

* filtervisualisation - look at the learned filters of a given model. In the presented notebook a Chebnet is trained with only one conv layer and the visualisation is made on one MNIST image and one unit dirac to visualise the filter response.

* laplacianprediction - look at the prediction error of a given model on different lattices / laplacians

