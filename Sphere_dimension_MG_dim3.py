import os
import sys
import time
import torch
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt

import models 
import utils
    
if len(sys.argv) > 1:
    num_rep = int(sys.argv[1])
    device_num = int(sys.argv[2])
else:
    num_rep = 0
    device_num = 0
if torch.cuda.device_count()>1:
    torch.cuda.set_device(device_num)
print('Id number: ', num_rep)


"""
Learn how to embed graph defined with points uniformly sampled from the (n-1)-dimensional sphere.
"""


#######################################
### Define parameters
#######################################
# Save
model_name = "Sphere_dimension_MG_tmp" # results will be saved in results/model_name
model_name_sampling = "Sphere_dimension" # folder where sampling points are saved

# Data generation
seed = 42 # seed parameter
Nrep = 8 # number of repetitions saved (only for final visualization)

# Training
Ntrain = 10000 # number of points sampled on the sphere
lr = 1e-3 # learning rate
batch_size = 32 # batch size
epochs = 20000 # number of epochs
# inDim = 10 # number of anchors
outDim = 5*3 # dimension of the output, number of mixtures x 3
Nlatent = 200 # dimension of latent layers
alpha = 1 # exponent in the distqnces
Ntest = 500 # training iterations between display
eps = 0 # to avoid nan values in acos


#######################################
### Prepare files and variables
#######################################
torch_type=torch.float
use_cuda=torch.cuda.is_available()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
print(device)
np.random.seed(seed+num_rep)
torch.manual_seed(seed+num_rep) 

if not os.path.exists("results/"+model_name):
    os.makedirs("results/"+model_name)
if not os.path.exists("results/"+model_name_sampling):
    os.makedirs("results/"+model_name_sampling)
if num_rep<0:
    train = False
else:
    train = True

#######################################
### Define the generator
#######################################
def generate_points_t(N,dim=3):
    out = torch.randn(N,dim,dtype=torch_type,device=device)
    out /= torch.linalg.norm(out,axis=1,keepdims=True)
    return out

#######################################
### Define the anchors
#######################################
if not(train):
    loss_train = []
    loss_test = []

Ndim_list = [3]
for Ndim in Ndim_list:
    K = 10000 # number of sampling points
    tau=1e-3
    inDim = Ndim+10
    if os.path.isfile("results/"+model_name_sampling+"/pts_sampled_"+str(inDim)+"_Ndim_"+str(Ndim)+".npz"):
        dat = np.load("results/"+model_name_sampling+"/pts_sampled_"+str(inDim)+"_Ndim_"+str(Ndim)+".npz")
        X_anchor = dat['sampledPts']
        inDim_ = dat['N']
        K_ = dat['K']
        tau_ = dat['tau']
        if ((tau_==tau) and (K_==K) and (inDim_==inDim)):
            print("Load anchors")
            X_anchor_t = torch.tensor(X_anchor).type(torch_type).to(device)
            inDim = X_anchor.shape[0]
        else:
            dir_save="results/"+model_name_sampling+"/pts_sampled_"+str(inDim)+".npz"
            X_anchor = utils.sample_convex(inDim, K, Ndim, tau, dir_save, verbose=True)
            X_anchor_t = torch.tensor(X_anchor).type(torch_type).to(device)
    else:
        dir_save="results/"+model_name_sampling+"/pts_sampled_"+str(inDim)+"_Ndim_"+str(Ndim)+".npz"
        X_anchor = utils.sample_convex(inDim, K, Ndim, tau, dir_save, verbose=True)
        X_anchor_t = torch.tensor(X_anchor).type(torch_type).to(device)


    #######################################
    ### Define network
    #######################################
    net_MG = models.MG2_transformer(inDim, outDim, N_latent=Nlatent, p=0., bn=False).to(device).train()
    # net_Euclidean = models.NetMLP(inDim, outDim//3, N_latent=Nlatent, p=0., bn=False).to(device).train()
    # checkpoint = torch.load("results/"+model_name_E+"/net_Ndim_"+str(Ndim)+"_"+str(num_rep)+".pt",map_location=device)
    # net_Euclidean.load_state_dict(checkpoint['model_state_dict'])
    # net_MG = models.MG2_transformer_extension_MLP(net_Euclidean).to(device).train()
    net_MG.summary()
    print("#parameters: {0}".format(sum(p.numel() for p in net_MG.parameters() if p.requires_grad)))

    #######################################
    ### Trainning
    #######################################
    # Prepare training
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(net_MG.parameters(), lr, weight_decay=5e-6)

    X = generate_points_t(Ntrain,dim=Ndim)
    dist_true_t = (torch.acos(torch.clamp(torch.matmul(X,X.transpose(1,0)),-1+eps,1-eps))/np.pi).fill_diagonal_(0)

    if train:
        loss_tot = []
        t0 = time.time()
        for ep in range(epochs):
            # step size decay
            if ep%1000==0 and ep!=0:
                for param_group in optimizer.param_groups:
                    param_group["lr"] = lr*(1-(1-0.0001)*ep/epochs)
           
            ind = np.random.randint(0,Ntrain,batch_size)
            input = (torch.acos(torch.clamp(torch.matmul(X[ind,:],X_anchor_t.transpose(1,0)),-1+eps,1-eps))/np.pi).requires_grad_(True)

            optimizer.zero_grad()
            out = net_MG(input)
            dist_mat_est = utils.dist_W2_MG_1D(out)
            loss = criterion(dist_true_t[ind,:][:,ind]**2**alpha,dist_mat_est)
            loss.backward()
            optimizer.step()

            loss_tot.append(loss.item())

            if ep%Ntest==0 and ep!=0:
                Xtest = generate_points_t(batch_size,dim=Ndim)
                input = torch.acos(torch.clamp(torch.matmul(Xtest,X_anchor_t.transpose(1,0)),-1+eps,1-eps))/np.pi
                dist_test_t = (torch.acos(torch.clamp(torch.matmul(Xtest,Xtest.transpose(1,0)),-1+eps,1-eps))/np.pi).fill_diagonal_(0)
                out = net_MG(input)
                dist_mat_test = utils.dist_W2_MG_1D(out)
                dist_val = criterion(dist_mat_test,dist_test_t**2**alpha)
                dist_max_val = torch.topk((torch.abs(dist_mat_test-dist_test_t**2**alpha)),1)[0].mean()
                distortion_val = (dist_mat_test/(dist_test_t**2**alpha)).fill_diagonal_(0)
                distortion_val = torch.max(distortion_val)
                abs_err = np.abs(dist_mat_test.detach().cpu().numpy()-dist_test_t.detach().cpu().numpy()**2).mean()
                print("Ndim: {0}, {1}/{2} -- Loss over iterations: {3:5.5} -- Loss new points {4:5.5} (avg max: {5:5.5}) -- abs err: {6:2.2} -- Distortion {7:2.2} -- Training time: {8:2.2}".format(Ndim,ep,epochs,np.mean(loss_tot[-Ntest:]),dist_val,dist_max_val,abs_err,distortion_val,time.time()-t0))

            if ep%(np.max([epochs//100,Ntest]))==0 and ep!=0:
                torch.save({
                    'iter': ep,
                    'model_state_dict': net_MG.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss_tot': loss_tot,
                    't_train': time.time()-t0,
                    }, "results/"+model_name+"/net_Ndim_"+str(Ndim)+"_"+str(num_rep)+".pt")

        torch.save({
            'iter': ep,
            'model_state_dict': net_MG.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss_tot': loss_tot,
            't_train': time.time()-t0,
            }, "results/"+model_name+"/net_Ndim_"+str(Ndim)+"_"+str(num_rep)+".pt")

    else:
        loss_train_ = []
        loss_test_ = []
        for num_rep in range(Nrep):
            checkpoint = torch.load("results/"+model_name+"/net_Ndim_"+str(Ndim)+"_"+str(num_rep)+".pt",map_location=device)
            loss_tot = checkpoint['loss_tot']

            # net_MG = DirectOptimization1D_NN_v2(inDim, outDim, N_latent=128, weights=True).to(device).train()
            net_MG.load_state_dict(checkpoint['model_state_dict'])
            net_MG = net_MG.eval()

            Xtest = generate_points_t(batch_size,dim=Ndim)
            input = torch.acos(torch.clamp(torch.matmul(Xtest,X_anchor_t.transpose(1,0)),-1+eps,1-eps))/np.pi
            dist_test_t = (torch.acos(torch.clamp(torch.matmul(Xtest,Xtest.transpose(1,0)),-1+eps,1-eps))/np.pi).fill_diagonal_(0)
            out = net_MG(input)
            dist_mat_test = utils.dist_W2_MG_1D(out)
            dist_val = torch.abs(dist_mat_test - dist_test_t**2)

            loss_train_.append(loss_tot[-1])
            loss_test_.append(dist_val.mean().detach().cpu().numpy())
        loss_train.append(np.mean(loss_train_))
        loss_test.append(np.mean(loss_test_))

if not(train):
    plt.figure(1)
    plt.plot(Ndim_list,loss_test)