import time
import torch
import faiss
import random
import cupy as cp
import numpy as np
import networkx as nx
import matplotlib as mpl
from geomloss import SamplesLoss 
import matplotlib.pyplot as plt; plt.ion()

####################################################
### Others
####################################################
def colorFader(c1,c2,mix=0): #fade (linear interpolate) from color c1 (at mix=0) to c2 (mix=1)
    c1=np.array(mpl.colors.to_rgb(c1))
    c2=np.array(mpl.colors.to_rgb(c2))
    return mpl.colors.to_hex((1-mix)*c1 + mix*c2)

####################################################
### Graph
####################################################
def hierarchy_pos(G, root=None, width=1., vert_gap = 0.2, vert_loc = 0, xcenter = 0.5):
    '''
    From Joel's answer at https://stackoverflow.com/a/29597209/2966723.  
    Licensed under Creative Commons Attribution-Share Alike 
    
    If the graph is a tree this will return the positions to plot this in a 
    hierarchical layout.
    
    G: the graph (must be a tree)
    
    root: the root node of current branch 
    - if the tree is directed and this is not given, 
      the root will be found and used
    - if the tree is directed and this is given, then 
      the positions will be just for the descendants of this node.
    - if the tree is undirected and not given, 
      then a random choice will be used.
    
    width: horizontal space allocated for this branch - avoids overlap with other branches
    
    vert_gap: gap between levels of hierarchy
    
    vert_loc: vertical location of root
    
    xcenter: horizontal location of root
    '''
    if not nx.is_tree(G):
        raise TypeError('cannot use hierarchy_pos on a graph that is not a tree')

    if root is None:
        if isinstance(G, nx.DiGraph):
            root = next(iter(nx.topological_sort(G)))  #allows back compatibility with nx version 1.11
        else:
            root = random.choice(list(G.nodes))

    def _hierarchy_pos(G, root, width=1., vert_gap = 0.2, vert_loc = 0, xcenter = 0.5, pos = None, parent = None):
        '''
        see hierarchy_pos docstring for most arguments

        pos: a dict saying where all nodes go if they have been assigned
        parent: parent of this branch. - only affects it if non-directed

        '''
    
        if pos is None:
            pos = {root:(xcenter,vert_loc)}
        else:
            pos[root] = (xcenter, vert_loc)
        children = list(G.neighbors(root))
        if not isinstance(G, nx.DiGraph) and parent is not None:
            children.remove(parent)  
        if len(children)!=0:
            dx = width/len(children) 
            nextx = xcenter - width/2 - dx/2
            for child in children:
                nextx += dx
                pos = _hierarchy_pos(G,child, width = dx, vert_gap = vert_gap, 
                                    vert_loc = vert_loc-vert_gap, xcenter=nextx,
                                    pos=pos, parent = root)
        return pos
            
    return _hierarchy_pos(G, root, width, vert_gap, vert_loc, xcenter)


####################################################
### Distances
####################################################
"""
Return l2 distance matrix of vector.
"""
def pairwise_distances(x):
    return torch.norm(x[:, None] - x, dim=2, p=2)

"""
Given a matrix A, it computes the quantity || A+A^t ||_2^2.
"""
def distance_matrix_add(dist_vec):
    x = dist_vec.permute(1,0).contiguous().view(-1,dist_vec.shape[0])
    inner = 2*torch.matmul(x.transpose(1, 0), x)
    xx = torch.sum(x**2, dim=0, keepdim=True)
    pairwise_distance = xx + inner + xx.transpose(1, 0)
    return pairwise_distance

"""
Given a matrix A, it computes the quantity || A-A^t ||_2^2.
"""
def distance_matrix(dist_vec):
    x = dist_vec.permute(1,0).contiguous().view(-1,dist_vec.shape[0])
    inner = -2*torch.matmul(x.transpose(1, 0), x)
    xx = torch.sum(x**2, dim=0, keepdim=True)
    pairwise_distance = xx + inner + xx.transpose(1, 0)
    return pairwise_distance

"""
Given two vectors/matrices v1 and v2, it computes the quantity || v1-v2^t ||_2^2.
"""
def distance_matrix_diff(v1,v2):
    inner = -2*torch.matmul(v1, v2.transpose(1, 0))
    xx1 = torch.sum(v1**2, dim=1, keepdim=True)
    xx2 = torch.sum(v2**2, dim=1, keepdim=True)
    pairwise_distance = xx1 + inner + xx2.transpose(1, 0)
    return pairwise_distance

"""
Distance matrix induced by the Fisher Information matrix between two Gaussian distribution.
dist1, dist2: size (batch,2), containing mean and variance of the 1d Gaussian distribution.

See https://arxiv.org/pdf/1210.2354.pdf, Equation (6).
"""
def dist_mat_Fisher_Rao(dist):
    dist2 = torch.cat((dist[:,0:1]/np.sqrt(2),-dist[:,1:2]),dim=1)
    dist1 = torch.cat((dist[:,0:1]/np.sqrt(2),dist[:,1:2]),dim=1)
    tmp1 = torch.linalg.norm(dist1[:,None,:]-dist2[None,:,:],dim=2)+torch.linalg.norm(dist1[:,None,:]-dist1[None,:,:],dim=2)
    tmp2 = torch.linalg.norm(dist1[:,None,:]-dist2[None,:,:],dim=2)-torch.linalg.norm(dist1[:,None,:]-dist1[None,:,:],dim=2)
    tmp = tmp1/(tmp2+1e-12)
    dist_out = np.sqrt(2)*torch.log(tmp)
    return dist_out

"""
Squared Wasserstein 2 distance between two 1-dimensional gaussian.
It corresponds to the l2 distance in this case.
"""
def dist_mat_W2(dist):
    d1 = distance_matrix(dist[:,0:1])
    d2 = distance_matrix(dist[:,1:2])
    return d1+d2

"""
Compute squared Wasserstein distance between two 1-dimensional 
Gaussian random variables.
"""
def GaussianW22_1D_t(m0,m1,sigma0,sigma1):
    ss0  = torch.sqrt(sigma0)
    s010 = torch.sqrt(ss0*sigma1*ss0)
    d = (m0-m1)**2+(sigma0+sigma1-2*s010)
    return d

"""
Approximate Wasserstein distance between real Gaussian mixtures based on the paper
"A Wasserstein-type distance in the space of Gaussian Mixture Models".

Input: 
 - GM1, GM2: Gaussian mixtures of size (batch, #mixture, 3). Last dimension contains 
             the mean, the standard deviation and the weight of the mixture.
"""
def GW2_1D_t(GM0, GM1,blur=0.05,scaling=0.5):
    batch_size, outDim, _ = GM0.shape
    distr0 = GM0[:,:,:2].view(batch_size,1,outDim,2).repeat(1,batch_size,1,1).view(batch_size*batch_size,outDim,2)
    distr1 = GM1[:,:,:2].view(1,batch_size,outDim,2).repeat(batch_size,1,1,1).view(batch_size*batch_size,outDim,2)
    pi0 = GM0[:,:,2].view(batch_size,1,outDim,1).repeat(1,batch_size,1,1).view(batch_size*batch_size,outDim)
    pi1 = GM1[:,:,2].view(1,batch_size,outDim,1).repeat(batch_size,1,1,1).view(batch_size*batch_size,outDim)
    Loss_sinkhorn = SamplesLoss("sinkhorn", p=2, blur=blur, scaling=scaling, backend="tensorized")
    distGW2 = Loss_sinkhorn(pi0,distr0*np.sqrt(2), pi1, distr1*np.sqrt(2))
    distGW2 = distGW2.view(batch_size,batch_size)    
    return distGW2

"""
Return the distance matrix between two Gaussian mixtures.
"""
def dist_W2_MG_1D(gauss_est,blur=0.05,scaling=0.5):
    distGW2 = GW2_1D_t(gauss_est,gauss_est,blur=0.05,scaling=0.5)
    return distGW2


####################################################
### Sampling
####################################################
# Functions from "Deep-Blur : Blind Identification and Deblurring with Convolutional Neural Networks".

"""
For X=(x1,...,xN) in NxD, computes the gradient of X -> sum_i min_j |xi-xj|
"""
def grad_maxisum_gpu(X):  
    X = cp.asnumpy(X)
    D = X.shape[1]
    index = faiss.IndexFlatL2(D) 
    res = faiss.StandardGpuResources()
    gpu_index_flat = faiss.index_cpu_to_gpu(res, 0, index) 
    gpu_index_flat.add(X.copy(order='C').astype("float32")) 
    dist, ind = gpu_index_flat.search(X.copy(order='C').astype("float32"), 2) 
    dif = X-X[ind[:, 1], :]
    dist = dist[:, 1]
    return cp.asarray(dif), cp.asarray(dist)

"""
For X=(x1,...,xN) in NxD, computes the gradient of X -> sum_i min_j |xi-xj|
"""
def grad_maxisum(X):
    X = cp.asnumpy(X)
    D = X.shape[1]
    index = faiss.IndexFlatL2(D) 
    index.add(X.copy(order='C').astype("float32")) 
    dist, ind = index.search(X.copy(order='C').astype("float32"), 2) 
    dif = X-X[ind[:, 1], :]
    dist = dist[:, 1]
    return cp.asarray(dif), cp.asarray(dist)

"""
Place N points on the n-sphere S^n as far as possible from each other.

INPUT:
  -N: number of points to sample
  -K: number of iterations
  -Ndim: dimension of the sphere
  -tau: step gradient descent
  -dir_save: where to save the output of this algorithm.

Adaptation of the code in "Deep-Blur: Blind Identification and Deblurring with
Convolutional Neural Networks".
"""
def sample_convex(N, K, Ndim=3, tau=1e-4, dir_save="pts_sampled.npz", verbose=False):  
  # Parameters
  np.random.seed(0)
  L = cp.random.randn(N, Ndim)/Ndim  # Starting convex combinations
  L =  L/np.linalg.norm(L,axis=1,keepdims=True)
  N = L.shape[0]
  X = L
  gradient = lambda X : grad_maxisum(X)
  t0 = time.time()
  for k in range(K):
      # Gradient computation
      X = L
      dif, dist = gradient(X)
      grad = dif/(dist[:, None] + 1e-8)
      grad = grad

      # Projected gradient descent
      L = L + tau*grad
      L =  L/np.linalg.norm(L,axis=1,keepdims=True)

      # Display
      if verbose:
        plt.ion()
        if k%500 == 0:
            X = L
            if Ndim >=3:
                fig = plt.figure(100)
                plt.clf()
                ax = fig.add_subplot(111, projection='3d')
                ax.scatter(X.get()[:, 0], X.get()[:, 1], X.get()[:, 2],'ob',s=100)

                yy = np.random.randn(1000,3)
                yy = yy/np.linalg.norm(yy,axis=1,keepdims=True)
                ax.scatter(yy[:, 0], yy[:, 1], yy[:, 2],'x',s=1)

                dif, dist = gradient(X) 
                t1 = time.time()
                plt.title('Iteration: %i/%i - tps:%1.2e - Min dist: %1.2e \n'%(k, K, t1-t0, np.min(dist[:N])))
                
                print('Iteration: %i/%i - tps:%1.2e - Min dist: %1.2e'%(k, K, t1-t0, np.min(dist[:N])))
                plt.pause(0.05)
            np.savez(dir_save,sampledPts=X,N=N,K=K,tau=tau)
        plt.ioff()

  if verbose:
    print('Iteration: %i/%i - tps:%1.2e - Min dist: %1.2e \n'%(k, K, time.time()-t0, np.min(dist)))
  return X

"""
Greedy sampling based on the distance matrix. First point is added to the list of 
fixed point, and then the point the farthest away from the already selected points.
"""
def greedy_sampling(inDim, dist):
    ptsFixed = np.zeros(inDim,dtype=int)
    dist_tree_tmp = dist.copy()
    for i in range(inDim):
        if i == 0:
            idx = 0
            ptsFixed[i] = idx
        else:
            idx = np.argmax(np.min(dist_tree_tmp[:,ptsFixed[:i]],1))
            ptsFixed[i] = idx
    return ptsFixed