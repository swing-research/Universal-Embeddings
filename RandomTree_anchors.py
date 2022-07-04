import os
import sys
import time
import torch
import numpy as np
import torch.nn as nn
import networkx as nx
import matplotlib.pyplot as plt
from networkx.drawing.nx_pydot import graphviz_layout

import models 
import utils


if len(sys.argv) > 1:
    num_rep = int(sys.argv[1])
    device_num = int(sys.argv[2])
else:
    num_rep = -1
    device_num = 0
if torch.cuda.device_count()>1:
    torch.cuda.set_device(device_num)
print('Id number: ', num_rep)

"""
Embedding of random tree using into Gaussian micture space.
Train with different number of anchors and see how it affects the embedding.
"""

# Save
model_name = "RandomTree_MG_anchors" # results will be saved in results/model_name

# Data generation
Npts = 30 # number of vertices
seed = 42 # seed parameter
NEval = 7 # number of testing points

# Training
lr = 1e-5 # learning rate
epochs = 10000 # number of epochs
outDim = 7*3 # dimension of the output, number of mixtures x 3
Nlatent = 512 # dimension of latent layers
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
if num_rep<0:
    train = False
else:
    train = True

#######################################
### Define the data
#######################################
# Generate tree
G = nx.dense_gnm_random_graph(n=Npts,m=80, seed=0)
idx_origin = np.arange(len(G.nodes))

# Display
fig = plt.figure(1)
fig.set_size_inches(22, 10.5)
plt.clf()
pos = graphviz_layout(G, prog="twopi")
nx.draw(G, pos=pos, with_labels=True,node_size=300)
plt.savefig("results/"+model_name+"/TreeTrue.png")

# Compute distance
dist_tree = np.zeros((Npts,Npts))
idx_origin = np.random.choice(idx_origin,len(idx_origin),replace=False)
for i in range(idx_origin.shape[0]):
    for j in range(idx_origin.shape[0]):
        dist_tree[i,j] = nx.dijkstra_path_length(G,idx_origin[i],idx_origin[j])
dist_tree /= dist_tree.max()
dist_tree_t = torch.tensor(dist_tree).type(torch_type).to(device)
idx_origin_t = torch.tensor(idx_origin).type(torch_type).to(device).view(-1,1)


#######################################
### Trainning
#######################################
## Loss function
criterion = nn.MSELoss()
## Iterate over number of mixture
inDim_list = np.array([3,5,7,9,12,15]) 
# inDim_list = np.array([3,5,7,12]) 
for inDim in inDim_list:
    np.random.seed(seed+num_rep)
    torch.manual_seed(seed+num_rep) 

    # Initialize fixed points
    ptsFixed = utils.greedy_sampling(inDim, dist_tree)
    ptsNonFixed = np.setdiff1d(np.arange(Npts),ptsFixed)
    ptsEval = np.random.choice(ptsNonFixed,NEval,replace=False)
    ptsTrain = np.setdiff1d(np.arange(Npts),ptsEval)
    input = dist_tree_t[ptsTrain][:,ptsFixed]
    input_full = dist_tree_t[:,ptsFixed]

    ## Define the model
    net_MG = models.MG2_transformer(inDim, outDim, N_latent=Nlatent, p=0.2, bn=False).to(device).train()
    net_MG.summary()
    print("#parameters: {0}".format(sum(p.numel() for p in net_MG.parameters() if p.requires_grad)))
    
    optimizer = torch.optim.Adam(net_MG.parameters(), lr, weight_decay=5e-6)
    
    loss_tot = []
    if train:
        try:
            t0 = time.time()
            for ep in range(epochs):
                # step size decay
                if ep%(np.max([epochs//1000,Ntest]))==0 and ep!=0:
                    for param_group in optimizer.param_groups:
                        param_group["lr"] = lr*(1-(1-0.1)*ep/epochs)

                optimizer.zero_grad()
                out = net_MG(input)
                dist_mat_est = utils.dist_W2_MG_1D(out)
                loss = criterion((dist_tree_t[ptsTrain,:][:,ptsTrain]**2)**alpha,dist_mat_est)
                loss.backward()
                optimizer.step()

                loss_tot.append(loss.item())

                if ep%Ntest==0 and ep!=0:
                    net_MG = net_MG.eval()
                    out = net_MG(input_full)
                    dist_mat_est = utils.dist_W2_MG_1D(out)
                    dist_val = np.mean(np.abs(dist_mat_est[ptsEval].detach().cpu().numpy()-dist_tree_t[ptsEval].detach().cpu().numpy()**2))
                    print("N rep: {0} -- N in {1} || {2}/{3} -- Loss over iterations: {4:5.5} -- Loss new points {5:5.5} -- Training time: {6:2.2}".format(num_rep,inDim,ep,epochs,np.mean(loss_tot[-Ntest:]),dist_val,time.time()-t0))
                    net_MG = net_MG.train()

                if ep%(np.max([epochs//10,Ntest]))==0 and ep!=0:
                    print("Save model")
                    torch.save({
                        'epoch': ep,
                        'model_state_dict': net_MG.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss_tot': loss_tot,
                        't_train': time.time()-t0,
                        }, "results/"+model_name+"/net_"+str(inDim)+"_"+str(num_rep)+".pt")

            torch.save({
                'epoch': ep,
                'model_state_dict': net_MG.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss_tot': loss_tot,
                't_train': time.time()-t0,
                }, "results/"+model_name+"/net_"+str(inDim)+"_"+str(num_rep)+".pt")
        except:
            print("##################################")
            print("###########  ERROR  ##############")
            print("##################################")
            print("N anchors {0} -- Id of repetition {1}".format(inDim, num_rep))
            continue


#######################################
### Display
#######################################
if not(train):
    Nrep = 8
    n_inDim_list = np.zeros((Nrep,len(inDim_list)))
    loss_list = np.zeros((Nrep,len(inDim_list)))
    loss_val_list = np.zeros((Nrep,len(inDim_list)))
    niter_list = np.zeros((Nrep,len(inDim_list)))

    for i, num_rep in enumerate(range(Nrep)):
        for j, inDim in enumerate(inDim_list):
            np.random.seed(seed)
            torch.manual_seed(seed) 

            # Initialize fixed points
            ptsFixed = utils.greedy_sampling(inDim, dist_tree)
            ptsNonFixed = np.setdiff1d(np.arange(Npts),ptsFixed)
            ptsEval = np.random.choice(ptsNonFixed,NEval,replace=False)
            ptsTrain = np.setdiff1d(np.arange(Npts),ptsEval)
            input = dist_tree_t[ptsTrain][:,ptsFixed]
            input_full = dist_tree_t[:,ptsFixed]

            # Define the model
            net_MG = models.MG2_transformer(inDim, outDim, N_latent=Nlatent, p=0., bn=False).to(device).train()
            net_MG.summary()
            print("#parameters: {0}".format(sum(p.numel() for p in net_MG.parameters() if p.requires_grad)))

            # Load data 
            checkpoint = torch.load("results/"+model_name+"/net_"+str(inDim)+"_"+str(num_rep)+".pt")
            loss_nn = checkpoint['loss_tot']
            net_MG.load_state_dict(checkpoint['model_state_dict'])
            net_MG = net_MG.eval()

            out = net_MG(input_full)
            dist_mat_est = utils.dist_W2_MG_1D(out)
            dd = np.abs(dist_mat_est[ptsEval].detach().cpu().numpy()-dist_tree_t[ptsEval].detach().cpu().numpy()**2)
            dist_val = np.mean(dd)
            dd = np.abs(dist_mat_est[ptsTrain].detach().cpu().numpy()-dist_tree_t[ptsTrain].detach().cpu().numpy()**2)
            dist_train = np.mean(dd)

            n_inDim_list[i,j] = inDim
            loss_list[i,j] = dist_train
            loss_val_list[i,j] = dist_val
            niter_list[i,j] = len(loss_nn)


    loss_val_list_ = np.zeros((Nrep-2,len(inDim_list)))
    loss_list_ = np.zeros((Nrep-2,len(inDim_list)))
    for k in range(len(inDim_list)):
        ind_ = (loss_val_list[:,k]>np.min(loss_val_list[:,k])) * (loss_val_list[:,k]<np.max(loss_val_list[:,k]))
        loss_val_list_[:,k] = loss_val_list[ind_,k]
        ind_ = (loss_list[:,k]>np.min(loss_list[:,k])) * (loss_list[:,k]<np.max(loss_list[:,k]))
        loss_list_[:,k] = loss_list[ind_,k]


    loss_val_list = loss_val_list_
    loss_list = loss_list_


    plt.figure(2)
    plt.clf()
    plt.plot(n_inDim_list[0,:],np.mean(loss_val_list,0),'r',label='Validation')
    plt.plot(n_inDim_list[0,:],np.mean(loss_list,0),'b',label='Train')
    plt.xlabel("# of anchors")
    plt.ylabel("Average abs. error")
    sig = 0.01
    plt.ylim([np.min([np.min(np.mean(loss_val_list,0)),np.min(np.mean(loss_list,0))])-sig,np.max([np.max(np.mean(loss_val_list,0)),np.max(np.mean(loss_list,0))])+sig])
    plt.xlim(n_inDim_list[0,0], n_inDim_list[0,-1])
    plt.grid(True)
    plt.legend()
    plt.savefig("results/"+model_name+"/loss_Nmixture.png")

    plt.figure(3)
    plt.clf()
    plt.plot(n_inDim_list[0,:],np.mean(niter_list,0),'k')
    plt.savefig("results/"+model_name+"/Niter_Nin.png")

    plt.figure(4)
    plt.clf()
    for ii in range(loss_val_list.shape[0]):
        plt.plot(n_inDim_list[ii],loss_val_list[ii],'r',label='Validation')
        plt.plot(n_inDim_list[ii],loss_list[ii],'b',label='Train')

    np.savetxt("results/"+model_name+"/losses.txt", (n_inDim_list[0,:],np.mean(loss_list,0), np.mean(loss_val_list,0)) )


    plt.figure(1)
    plt.clf()
    pos = graphviz_layout(G, prog="twopi")
    nx.draw(G, pos, node_size=120, node_color="#09a433",alpha =0.9, width=1)
    nx.draw_networkx_nodes(G, pos=pos, node_size=120, nodelist=idx_origin[ptsFixed], node_color="#000000",alpha =0.9)
    nx.draw_networkx_nodes(G, pos=pos, node_size=120, nodelist=idx_origin[ptsEval], node_color="#e5e0e0",alpha =0.9)
    plt.savefig("results/"+model_name+"/TreeTrue.png")
