# Universal embeddings
Repository for 'Probabilistic Graph Transformers Are Universal Feature Maps Capable of Representing any Finite Dataset' [(paper)](https://arxiv.org/)

These codes were used to generate illustrations of the paper. 

## Abstract
The problem of representing a finite dataset equipped with a dissimilarity metric using a trainable
feature map is one of the basic questions in machine learning. A hallmark of deep learning models
is the capacity of a model’s hidden layers to efficiently encode discrete Euclidean data into low-
dimensional Euclidean feature spaces, from which their last layer can readout predictions. How-
ever, when the finite dataset is not Euclidean, it has been empirically confirmed that representations
in non-Euclidean spaces systematically outperform traditional Euclidean feature representations.
In this paper, we prove that given any n-point metric space X there exists a probabilistic graph
transformer (PGT) which can bi-α-Hölder embed X into univariate Gaussian mixtures MG(R) of
Delon and Desolneux (2020) with small distortion to X ’s metric for any α ∈ (1/2, 1). Moreover,
this PGT’s depth and width are approximately linear in n. We then show that, for any “distortion
level” D > 2, there is a PGT which can represent X in MG(R) such that any uniformly sam-
pled pair of points in X are bi-Lipschitz embedded with distortion at-most D2, with probability
O(n^{−4e/D} ). We show that if X has a suitable geometric prior (e.g. X is a combinatorial tree or
a finite subspace of a suitable Riemann manifold), then the PGT architecture can deterministically
bi-Lipschitz embed X into MG(R) with low metric distortion. As applications, we consider PGT
embeddings of 2-Hop combinatorial graphs (such as friendship graphs, cocktail graphs, complete
bipartite, etc...), trees, and n-point subsets of Riemannian manifolds.


## Requierements
All experiments have been conducted using Python 3 using mainly Pytorch and Numpy librairies. We also requieres an optimal transport library. To that end, we use the fast implementation provided in [geomloss](https://www.kernel-operations.io/geomloss/). 

Finally, one need the following libraries:

- numpy
- torch
- geomloss
- time

For the experiment about the n-dimensional sphere, we also use:

- faiss 
- cupy 

For visualization: 

- plotly (3D plots)
- networkx (graph plots)
- matplotlib

## Embedding of binary tree
Reproduce the illustartions of paragraph 4.1, "Embedding of metric tree". We first need to train the three different embeddings.

#### Trainning
To train the neural network to embed into Euclidean space, run the following script:

```console
python Tree_Euclidean.py

```

To train the neural network to embed into Hyperbolic space (single Gaussian measure), run the following script:

```console
python Tree_Hyperbolic.py

```

To train the neural network to embed into the space of Gaussian mixtures, run the following script:

```console
python Tree_MG.py

```

#### Comparisons
Finally, to combine all the results and generates the figures, use the folloying commad:

```console
python Tree_compare.py

```

## Embedding of radom tree 
Reproduce the illustartions of paragraph 4.2, "Embedding a random graph: what parameters matter?". 

### Importance of number of mixtures
#### Trainning
Run the following bash script to train the Probabilistic Graph Transformers with the different number of mixtures. The same network is trained for 8 different seed parameters, on at most 4 GPUS. 

```console
bash batch_RandomTree_mixtures.sh

```

#### Evaluation
Run the following command to generate the figures based on the previously trained networks. The figure is obtained by averaging the results of 8 different trainning with different seed parameters.

```console
python RandomTree_mixtures.py

```

### Importance of number of anchors
#### Trainning
Run the following bash script to train the Probabilistic Graph Transformers with the different parameters. The same network is trained for 8 different seed parameters, on at most 4 GPUS. 

```console
bash batch_RandomTree_anchors.sh

```

#### Evaluation
Run the following command to generate the figures based on the previously trained networks. The figure is obtained by averaging the results of 8 different trainning with different seed parameters.

```console
python RandomTree_anchors.py

```

## Embedding of n-dimensional sphere 

#### Trainning

We first train a neural network to embed points from the n-dimensional sphere to the Euclidean space. The same network is trained for 8 different seed parameters, on at most 4 GPUS. 


```console
bash batch_Sphere_Euclidean.sh

```

We also train our Probabilistic Graph Transformer to embed the n-dimensional sphere into the space of Gaussian mixtures. This is done on several GPUS (if several available) using the commad:

```console
bash batch_Sphere_MG.sh

```

#### Evaluation
Finally, we display and compare results by calling the following python script:

```console
python Sphere_dimension_compare.py

``` 