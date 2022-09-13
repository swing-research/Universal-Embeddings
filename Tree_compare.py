import os
import torch
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from scipy.sparse.linalg import svds
from networkx.drawing.nx_pydot import graphviz_layout
from mpl_toolkits.axisartist.axislines import SubplotZero

import data_generator
import models 
import utils

if torch.cuda.device_count()>1:
    torch.cuda.set_device(0)


"""
Collect results about the embedding of tree.
Compare enbedding in Hyperbolic, Euclidean and Gaussian mixture spaces.
"""

# Save and load
model_name1 = "Tree_MG"
model_name2 = "Tree_Euclidean"
model_name3 = "Tree_Hyperbolic"
save_name = "Compare_Tree"

# Data generation
Nlevel = 6 # number of tree level
Nrep = 2 # number of leaves per node
seed = 42 # seed parameter
Ntrain = 111 # number of training points
inDim = 20 # number of anchors
load_sampler = False # load tree (save few minutes)


#######################################
### Prepare files and variables
#######################################
torch_type=torch.float
use_cuda=torch.cuda.is_available()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
print(device)
np.random.seed(seed)
torch.manual_seed(seed)

if not os.path.exists("results/"+save_name):
    os.makedirs("results/"+save_name)


#######################################
### Define the data
#######################################
# Generate tree
# G, dist_tree, idx_origin = data_generator.tree(Nlevel,Nrep,seed)
# Npts = dist_tree.shape[0]
if not(load_sampler):
    # Generate tree
    G, dist_tree, idx_origin = data_generator.tree(Nlevel,Nrep,seed)
    np.savez("results/"+model_name1+"/tree.npz",G=G,dist_tree=dist_tree,idx_origin=idx_origin)
else:
    dat = np.load("results/"+model_name1+"/tree.npz")
    G = dat['G']
    dist_tree = dat['dist_tree']
    idx_origin = dat['idx_origin']

# Compute distance matrix
dist_tree_t = torch.tensor(dist_tree).type(torch_type).to(device)
idx_origin_t = torch.tensor(idx_origin).type(torch_type).to(device).view(-1,1)

# Initialize fixed points
ptsFixed = utils.greedy_sampling(inDim, dist_tree)
input = dist_tree_t[:Ntrain,ptsFixed]
input_full = dist_tree_t[:,ptsFixed]


#######################################
### Load model
#######################################
# Model 1: Gaussian mixture
outDim = 5*3 # dimension of the output, number of mixtures x 3
Nlatent = 32 # dimension of latent layers
net_MG = models.MG2_transformer(inDim, outDim, N_latent=Nlatent, p=0., bn=False).to(device)
checkpoint = torch.load("results/"+model_name1+"/net.pt",map_location=device)
net_MG.load_state_dict(checkpoint['model_state_dict'])
net_MG = net_MG.eval()
net_MG = net_MG.train(False)
loss1 = checkpoint['loss_tot']
net_MG.summary()
print("MG model: #parameters: {0}".format(sum(p.numel() for p in net_MG.parameters() if p.requires_grad)))

# Model 2: Euclidean Gauss
outDim = 5*3 # dimension of the output, number of mixtures x 3
Nlatent = 32 # dimension of latent layers
net_Euclidean = models.NetMLP(inDim, outDim, N_latent=Nlatent, p=0., bn=False).to(device)
checkpoint = torch.load("results/"+model_name2+"/net.pt",map_location=device)
net_Euclidean.load_state_dict(checkpoint['model_state_dict'])
net_Euclidean = net_Euclidean.eval()
net_Euclidean = net_Euclidean.train(False)
loss2 = checkpoint['loss_tot']
net_Euclidean.summary()
print("Euclidean model: #parameters: {0}".format(sum(p.numel() for p in net_Euclidean.parameters() if p.requires_grad)))

# Model 3: 1D 
outDim = 2 # dimension of the output
Nlatent = 32 # dimension of latent layers
net_Hyperbolic = models.MG2_transformer(inDim, outDim, N_latent=Nlatent, weights=False, p=0., bn=False).to(device)
checkpoint = torch.load("results/"+model_name3+"/net.pt",map_location=device)
net_Hyperbolic.load_state_dict(checkpoint['model_state_dict'])
net_Hyperbolic = net_Hyperbolic.eval()
net_Hyperbolic = net_Hyperbolic.train(False)
loss3 = checkpoint['loss_tot']
net_Hyperbolic.summary()
print("Hyperbolic model: #parameters: {0}".format(sum(p.numel() for p in net_Hyperbolic.parameters() if p.requires_grad)))



#######################################
### Load model
#######################################
# Loss
st = 20
plt.figure(2)
plt.clf()
plt.plot(np.array(loss1[st:]),label='Gaussian mixture')
plt.plot(np.array(loss2[st:]),label='Euclidean')
plt.plot(np.array(loss3[st:]),label='Gaussian')
plt.legend()
plt.savefig("results/"+save_name+"/cf.png")

# Distance matrix
out1 = net_MG(input_full)
dist_mat_est1 = utils.dist_W2_MG_1D(out1)
err1 = np.abs(dist_mat_est1.detach().cpu().numpy()-dist_tree_t.detach().cpu().numpy()**2)
diff_mat1 = np.log(err1+1e-12)

out2 = net_Euclidean(input_full)
dist_mat_est2 = utils.distance_matrix(out2)
err2 = np.abs(dist_mat_est2.detach().cpu().numpy()-dist_tree_t.detach().cpu().numpy()**2)
diff_mat2 = np.log(err2+1e-12)

out3 = net_Hyperbolic(input_full)
# dist_mat_est3 = utils.dist_mat_Fisher_Rao(out3)**2
dist_mat_est3 = utils.distance_hyperbolic(out3)**2
err3 = np.abs(dist_mat_est3.detach().cpu().numpy()-dist_tree_t.detach().cpu().numpy()**2)
diff_mat3 = np.log(err3+1e-12)

fig = plt.figure(3)
plt.clf()
plt.imshow(diff_mat1,vmin=-7,cmap='jet')
plt.colorbar()
plt.clim(-1,-7)  
plt.savefig("results/"+save_name+"/distance_diff_MG.png")
plt.title('Difference dist mat -- MG')

fig = plt.figure(4)
plt.clf()
plt.imshow(diff_mat2,vmin=-7,cmap='jet')
plt.colorbar()
plt.clim(-1,-7)  
plt.savefig("results/"+save_name+"/distance_diff_Euclidean.png")
plt.title('Difference dist mat -- Euclidean')

fig = plt.figure(5)
plt.clf()
plt.imshow(diff_mat3,vmin=-7,cmap='jet')
plt.colorbar()
plt.clim(-1,-7)  
plt.savefig("results/"+save_name+"/distance_diff_Hyperbolic.png")
plt.title('Difference dist mat -- Hyperbolic')


print("On trainning points")
print("Mean: MG {0} -- Euclidean {1} -- Hyperbolic {2}".format(err1[:Ntrain,:Ntrain].mean(),err2[:Ntrain,:Ntrain].mean(),err3[:Ntrain,:Ntrain].mean()))
print("Max: MG {0} -- Euclidean {1} -- Hyperbolic {2}".format(err1[:Ntrain,:Ntrain].max(),err2[:Ntrain,:Ntrain].max(),err3[:Ntrain,:Ntrain].max()))

print("On new points")
print("Mean: MG {0} -- Euclidean {1} -- Hyperbolic {2}".format(err1[Ntrain:,Ntrain:].mean(),err2[Ntrain:,Ntrain:].mean(),err3[Ntrain:,Ntrain:].mean()))
print("Max: MG {0} -- Euclidean {1} -- Hyperbolic {2}".format(err1[Ntrain:,Ntrain:].max(),err2[Ntrain:,Ntrain:].max(),err3[Ntrain:,Ntrain:].max()))

print("On all")
print("Mean: MG {0} -- Euclidean {1} -- Hyperbolic {2}".format(err1.mean(),err2.mean(),err3.mean()))
print("Max: MG {0} -- Euclidean {1} -- Hyperbolic {2}".format(err1.max(),err2.max(),err3.max()))

np.savetxt("results/"+save_name+"/losses.txt",
    ((err1[:Ntrain,:Ntrain].mean(),err2[:Ntrain,:Ntrain].mean(),err3[:Ntrain,:Ntrain].mean()),
    (err1[:Ntrain,:Ntrain].max(),err2[:Ntrain,:Ntrain].max(),err3[:Ntrain,:Ntrain].max()),
    (err1[Ntrain:,Ntrain:].mean(),err2[Ntrain:,Ntrain:].mean(),err3[Ntrain:,Ntrain:].mean()),
    (err1[Ntrain:,Ntrain:].max(),err2[Ntrain:,Ntrain:].max(),err3[Ntrain:,Ntrain:].max())),fmt='%.3f',delimiter='---')



## Display tree with path to explore
path_explore = nx.shortest_path(G,87,117)
# path_explore = nx.shortest_path(G,351,471)

c1='#003eff' #blue
c2='#ff0080' #green
Nfr = len(path_explore)
col_gen = lambda i: utils.colorFader(c1,c2,np.linspace(0,1,Nfr)[i])
col_list1 = [col_gen(i) for i in np.arange(Nfr)]

sz= 120
plt.figure(6)
plt.clf()
pos = graphviz_layout(G, prog="twopi")
nx.draw(G, pos, node_size=sz, node_color="#09a433",alpha =0.9, width=2)
nx.draw_networkx_nodes(G, pos=pos, node_size=sz, nodelist=idx_origin[ptsFixed], node_color="#000000",alpha =0.9)
nx.draw_networkx_nodes(G, pos=pos, node_size=sz, nodelist=idx_origin[Ntrain:], node_color="#e5e0e0",alpha =0.9)
nx.draw_networkx_nodes(G, pos=pos, node_size=sz*2, nodelist=path_explore, node_color=col_list1,alpha =0.9)
plt.savefig("results/"+save_name+"/TreeTrue.png")
plt.savefig("results/"+save_name+"/TreeTrue.pdf")

# Extract path
list_path_out = []
for k in range(len(path_explore)):
    ii = np.where(idx_origin == path_explore[k])[0][0]
    list_path_out.append(ii)
Ndiscr = 100

# Compute distribution for MG
out_tmp = out1[list_path_out].detach().cpu().numpy()
sh = -0.5
sc = 1.8
x_discr =  sc*np.linspace(-1,1,Ndiscr)*(out_tmp[:,0].max()-out_tmp[:,0].min())+out_tmp[:,0].min()+sh
plt.rcParams['xtick.labelsize']=20
plt.rcParams['ytick.labelsize']=20
fig = plt.figure(7)
fig.clf()
ax = SubplotZero(fig, 111)
fig.add_subplot(ax)
for direction in ["xzero", "yzero"]:
    ax.axis[direction].set_axisline_style("-|>")
    ax.axis[direction].set_visible(True)
for direction in ["left", "right", "bottom", "top"]:
    # hides borders
    ax.axis[direction].set_visible(False)
ax.get_xaxis().set_ticks(np.arange(20,120,20)/100)
ax.get_yaxis().set_ticks(np.arange(0,4,1)/4)
fs = lambda x: np.log10(x+2.1)
for k in range(len(path_explore)):
    gg = np.zeros_like(x_discr)
    for j in range(out_tmp.shape[1]):
        gg += out_tmp[k,j,2]*np.exp( -(x_discr-out_tmp[k,j,0])**2/(2*fs(out_tmp[k,j,1])**2) ) / (np.sqrt(2*np.pi)*fs(out_tmp[k,j,1]))
    ax.plot(np.linspace(0,1,Ndiscr),gg,color=col_list1[k],linewidth=2)
plt.savefig("results/"+save_name+"/distr_MG.pdf")

# Euclidean
out_tmp = out2[list_path_out].detach().cpu().numpy()
np.random.seed(42)
uu, ss, vv = svds(out_tmp,k=2)
plt.rcParams['xtick.labelsize']=20
plt.rcParams['ytick.labelsize']=20
fig = plt.figure(8)
fig.clf()
shx =  -0.5*(uu[:,0].max()+uu[:,0].min())
shy =  -0.5*(uu[:,1].max()+uu[:,1].min())
ax.get_xaxis().set_ticks([])
ax.get_yaxis().set_ticks([])
ax = SubplotZero(fig, 111)
fig.add_subplot(ax)
ax.get_xaxis().set_ticks([])
ax.get_yaxis().set_ticks([])
for direction in ["xzero", "yzero"]:
    ax.axis[direction].set_axisline_style("-|>")
    ax.axis[direction].set_visible(True)
for direction in ["left", "right", "bottom", "top"]:
    # hides borders
    ax.axis[direction].set_visible(False)
ax.scatter(uu[:,0]+shx,uu[:,1]+shy,marker='o',c=col_list1)
plt.savefig("results/"+save_name+"/distr_Euclidean.pdf")

# Hyperbolic
out_tmp = out3[list_path_out].detach().cpu().numpy()
sh = -0.01
shx = 0.15
sc = 15
x_discr = sc*np.linspace(-1,1,Ndiscr)*(out_tmp[:,0].max()-out_tmp[:,0].min())+out_tmp[:,0].min()+sh
plt.rcParams['xtick.labelsize']=20
plt.rcParams['ytick.labelsize']=20
fig = plt.figure(9)
fig.clf()
ax = SubplotZero(fig, 111)
fig.add_subplot(ax)
for direction in ["xzero", "yzero"]:
    ax.axis[direction].set_axisline_style("-|>")
    ax.axis[direction].set_visible(True)
for direction in ["left", "right", "bottom", "top"]:
    # hides borders
    ax.axis[direction].set_visible(False)
for k in range(len(path_explore)):
    gg = np.exp( -(x_discr-out_tmp[k,0])**2/(2*out_tmp[k,1]**2) ) / (2*np.pi*np.sqrt(out_tmp[k,1]**2))
    ax.plot(np.linspace(0,1,Ndiscr),gg,color=col_list1[k],linewidth=2)
ax.get_xaxis().set_ticks(ax.get_xaxis().get_ticklocs()[2:-1])
ax.get_yaxis().set_ticks(ax.get_yaxis().get_ticklocs()[2:-1])
plt.savefig("results/"+save_name+"/distr_Hyperbolic.pdf")
