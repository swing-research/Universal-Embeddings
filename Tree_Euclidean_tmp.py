import os
import time
import torch
import numpy as np
import torch.nn as nn
import networkx as nx
import matplotlib.pyplot as plt

import data_generator
import models 
import utils

if torch.cuda.device_count()>1:
    torch.cuda.set_device(1)


"""
Illustrate the embedding into Euclidean space of a tree using the proposed probabilistic transformer.
"""

# Save
model_name = "Tree_Euclidean_tmp" # results will be saved in results/model_name

# Data generation
Nlevel = 8 # number of tree level
Nrep = 2 # number of leaves per node
seed = 42 # seed parameter
Ntrain = 400 # number of training points
inDim = 20 # number of anchors

# Training
load_sampler = True # load tree (save few minutes)
train = True # train the model
load_model = False # load previously trained model
lr = 1e-4 # learning rate
epochs = 200000 # number of epochs
outDim = 5*3 # dimension of the output, number of mixtures x 3
Nlatent = 50 # dimension of latent layers
alpha = 1 # exponent in the distqnces
Ntest = 500 # training iterations between display


#######################################
### Prepare files and variables
#######################################
torch_type=torch.float
use_cuda=torch.cuda.is_available()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
print(device)
np.random.seed(seed)
torch.manual_seed(seed)

if not os.path.exists("results/"+model_name):
    os.makedirs("results/"+model_name)

#######################################
### Define the data
#######################################
# Generate tree
if not(load_sampler):
    # Generate tree
    G, dist_tree, idx_origin = data_generator.tree(Nlevel,Nrep,seed)
    np.savez("results/"+model_name+"/tree.npz",G=G,dist_tree=dist_tree,idx_origin=idx_origin)
else:
    dat = np.load("results/"+model_name+"/tree.npz")
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

# # Display
# print(nx.forest_str(G, sources=[0]))
# plt.figure(1)
# pos = utils.hierarchy_pos(G,0)
# nx.draw(G, pos=pos, with_labels=True)
# plt.savefig("results/"+model_name+"/TreeTrue.png")


#######################################
### Trainning
#######################################
## Define the model
net_Euclidean = models.NetMLP(inDim, outDim, N_latent=Nlatent, p=0., bn=False).to(device).train()
net_Euclidean.summary()
print("#parameters: {0}".format(sum(p.numel() for p in net_Euclidean.parameters() if p.requires_grad)))

# Load previous model
if load_model and os.path.isfile("results/"+model_name+"/net.pt"):
    checkpoint = torch.load("results/"+model_name+"/net.pt",map_location=device)
    net_Euclidean.load_state_dict(checkpoint['model_state_dict'])
    net_Euclidean.train()

# Prepare training
optimizer = torch.optim.Adam(net_Euclidean.parameters(), lr, weight_decay=5e-6)
loss_tot = []
idx_train = np.arange(Ntrain)
idx_train_t = torch.tensor(idx_train).type(torch_type).to(device).view(-1,1)
d_true = dist_tree_t[idx_train,:][:,idx_train].detach().cpu().numpy()
criterion = nn.MSELoss()

if train:
    t0 = time.time()
    for ep in range(epochs):
        # step size decay
        if ep%(np.max([epochs//100,Ntest]))==0 and ep!=0:
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr*(1-(1-0.1)*ep/epochs)

        optimizer.zero_grad()
        out = net_Euclidean(input)
        dist_mat_est = utils.distance_matrix(out)
        loss = criterion((dist_tree_t[idx_train,:][:,idx_train]**2)**alpha,dist_mat_est)
        loss.backward()
        optimizer.step()
        loss_tot.append(loss.item())
        
        if ep%Ntest==0 and ep!=0:
            out = net_Euclidean(input_full)[Ntrain:]
            dist_mat_est = utils.distance_matrix(out)
            dist_val = criterion((dist_tree_t[Ntrain:,:][:,Ntrain:]**2)**alpha,dist_mat_est)
            dist_max_val = torch.topk((torch.abs(dist_mat_est-(dist_tree_t[Ntrain:,:][:,Ntrain:]**2)**alpha)),1)[0].mean()
            print("{0}/{1} -- Loss over iterations: {2} -- Loss validation {3} -- (avg max {4})".format(ep,epochs,np.mean(loss_tot[-Ntest:]),dist_val,dist_max_val))
            print("Time per 100 epochs: {0}".format(100*(time.time()-t0)/Ntest))

        if ep%(np.max([epochs//100,Ntest]))==0 and ep!=0:
            print("Save model")
            torch.save({
                'epoch': ep,
                'model_state_dict': net_Euclidean.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss_tot': loss_tot,
                }, "results/"+model_name+"/net.pt")
        t0 = time.time()

    torch.save({
        'epoch': ep,
        'model_state_dict': net_Euclidean.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss_tot': loss_tot,
        }, "results/"+model_name+"/net.pt")


#######################################
### Evaluation
#######################################
# Load model in case training was done previously
checkpoint = torch.load("results/"+model_name+"/net.pt",map_location=device)
net_Euclidean.load_state_dict(checkpoint['model_state_dict'])
net_Euclidean = net_Euclidean.eval()
loss_tot = checkpoint['loss_tot']

# Display training output
plt.figure(2)
plt.clf()
plt.plot(np.array(loss_tot[20:]))
plt.savefig("results/"+model_name+"/cf.png")

fig = plt.figure(3)
plt.clf()
out = net_Euclidean(input_full)
dist_mat_est = utils.distance_matrix(out)
diff_mat= np.log(np.abs(dist_mat_est.detach().cpu().numpy()-dist_tree_t.detach().cpu().numpy()**2)+1e-12)
plt.imshow(diff_mat,vmin=-7,cmap='jet')
plt.colorbar()
plt.clim(-1,-7)  
plt.savefig("results/"+model_name+"/distance_diff.png")
plt.title('Difference between matrix true')

plt.show()