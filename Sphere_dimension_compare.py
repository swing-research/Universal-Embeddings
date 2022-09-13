import os
import time
import torch
import numpy as np
import torch.nn as nn
import networkx as nx
from matplotlib import animation
import matplotlib.pyplot as plt

import models 
import utils

if torch.cuda.device_count()>1:
    torch.cuda.set_device(0)

"""
Script to generate plots regarding the embedding of the sphere.
This script has not been clean and commented properly. We still decide to keep it in the Github repo
since function inside are particularly usefull and difficult to use.
"""

#######################################
### Define parameters
#######################################
# # Save
model_name_E = "Sphere_dimension_Euclidean" # results will be saved in results/model_name
model_name_MG = "Sphere_dimension_MG" # results will be saved in results/model_name
model_name_Hyperbolic = "Sphere_dimension_Hyperbolic" # results will be saved in results/model_name
model_name_sampling = "Sphere_dimension" # folder where sampling points are saved
model_name = "Sphere_dimension" # folder where to save results

# Data generation
seed = 42 # seed parameter
Nrep = 4  # number of repetitions saved (only for final visualization)

# Training
batch_size = 2048 # batch size
outDim = 5*3 # dimension of the output, number of mixtures x 3
Nlatent = 200 # dimension of latent layers
alpha = 1 # exponent in the distqnces
eps = 0 # to avoid nan values in acos

#######################################
### Prepare files and variables
#######################################
torch_type=torch.float
use_cuda=torch.cuda.is_available()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
print(device)
np.random.seed(seed)
torch.manual_seed(seed) 

#######################################
### Evaluation
#######################################
def generate_points_t(N,dim=3):
    out = torch.randn(N,dim,dtype=torch_type,device=device)
    out /= torch.linalg.norm(out,axis=1,keepdims=True)
    return out

Ndim_list = np.arange(15,50,4)
loss_train_MG = np.zeros((Nrep,len(Ndim_list)))
loss_train_E = np.zeros((Nrep,len(Ndim_list)))
loss_train_Hyperbolic = np.zeros((Nrep,len(Ndim_list)))
loss_test_MG = np.zeros((Nrep,len(Ndim_list)))
loss_test_E = np.zeros((Nrep,len(Ndim_list)))
loss_test_Hyperbolic = np.zeros((Nrep,len(Ndim_list)))
t_train_MG = np.zeros((Nrep,len(Ndim_list)))
t_train_E = np.zeros((Nrep,len(Ndim_list)))
t_train_Hyperbolic = np.zeros((Nrep,len(Ndim_list)))
t_test_MG = np.zeros((Nrep,len(Ndim_list)))
t_test_E = np.zeros((Nrep,len(Ndim_list)))
t_test_Hyperbolic = np.zeros((Nrep,len(Ndim_list)))
dist_mat_err_MG = np.zeros((Nrep,len(Ndim_list),batch_size,batch_size))
dist_mat_err_E = np.zeros((Nrep,len(Ndim_list),batch_size,batch_size))
dist_mat_err_Hyperbolic = np.zeros((Nrep,len(Ndim_list),batch_size,batch_size))
dist_mat_MG = np.zeros((Nrep,len(Ndim_list),batch_size,batch_size))
dist_mat_E = np.zeros((Nrep,len(Ndim_list),batch_size,batch_size))
dist_mat_Hyperbolic = np.zeros((Nrep,len(Ndim_list),batch_size,batch_size))
dist_mat = np.zeros((Nrep,len(Ndim_list),batch_size,batch_size))
for j, n_rep in enumerate(range(Nrep)):
    for i, Ndim in enumerate(Ndim_list):
        inDim = Ndim+10
        #######################################
        ### Define the anchors
        #######################################
        K = 10000 # number of sampling points
        tau=1e-3
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
        # load model
        net_MG = models.MG2_transformer(inDim, outDim, N_latent=Nlatent, p=0., bn=False).to(device).train()
        net_MG.summary()
        print("#parameters: {0}".format(sum(p.numel() for p in net_MG.parameters() if p.requires_grad)))

        net_E = models.NetMLP(inDim, outDim, N_latent=Nlatent, p=0., bn=False).to(device).train()
        net_E.summary()
        print("#parameters Euclidean: {0}".format(sum(p.numel() for p in net_E.parameters() if p.requires_grad)))

        net_Hyperbolic = models.NetMLP(inDim, outDim, N_latent=Nlatent, p=0., bn=False, hyperbolic=True).to(device).train()
        net_Hyperbolic.summary()
        print("#parameters Hyperbolic: {0}".format(sum(p.numel() for p in net_Hyperbolic.parameters() if p.requires_grad)))

        # load weights
        checkpoint = torch.load("results/"+model_name_MG+"/net_Ndim_"+str(Ndim)+"_"+str(n_rep)+".pt",map_location=device)
        net_MG.load_state_dict(checkpoint['model_state_dict'])
        net_MG = net_MG.eval()
        net_MG = net_MG.train(False)

        checkpoint = torch.load("results/"+model_name_E+"/net_Ndim_"+str(Ndim)+"_"+str(n_rep)+".pt",map_location=device)
        net_E.load_state_dict(checkpoint['model_state_dict'])
        net_E = net_E.eval()
        net_E = net_E.train(False)

        checkpoint = torch.load("results/"+model_name_Hyperbolic+"/net_Ndim_"+str(Ndim)+"_"+str(n_rep)+".pt",map_location=device)
        net_Hyperbolic.load_state_dict(checkpoint['model_state_dict'])
        net_Hyperbolic = net_Hyperbolic.eval()
        net_Hyperbolic = net_Hyperbolic.train(False)

        #######################################
        ### Load data
        #######################################
        checkpoint = torch.load("results/"+model_name_MG+"/net_Ndim_"+str(Ndim)+"_"+str(n_rep)+".pt",map_location=device)
        loss_tot = checkpoint['loss_tot']
        t_train = checkpoint['t_train']
        loss_train_MG[j,i] = np.mean(loss_tot)
        t_train_MG[j,i] = t_train

        checkpoint = torch.load("results/"+model_name_E+"/net_Ndim_"+str(Ndim)+"_"+str(n_rep)+".pt",map_location=device)
        loss_tot = checkpoint['loss_tot']
        t_train = checkpoint['t_train']
        loss_train_E[j,i] = np.mean(loss_tot)
        t_train_E[j,i] = t_train

        checkpoint = torch.load("results/"+model_name_Hyperbolic+"/net_Ndim_"+str(Ndim)+"_"+str(n_rep)+".pt",map_location=device)
        loss_tot = checkpoint['loss_tot']
        t_train = checkpoint['t_train']
        loss_train_Hyperbolic[j,i] = np.mean(loss_tot)
        t_train_Hyperbolic[j,i] = t_train

        #######################################
        ### Evaluate data
        #######################################
        X = generate_points_t(batch_size,dim=Ndim)
        input = torch.acos(torch.clamp(torch.matmul(X,X_anchor_t.transpose(1,0)),-1+eps,1-eps))/np.pi
        dist_true_t = (torch.acos(torch.clamp(torch.matmul(X,X.transpose(1,0)),-1+eps,1-eps))/np.pi).fill_diagonal_(0)
        criterion = nn.MSELoss()

        t0 = time.time()
        out = net_MG(input)
        t1 = time.time()
        dist_est = utils.dist_W2_MG_1D(out)
        err = np.abs(dist_est.detach().cpu().numpy()-dist_true_t.detach().cpu().numpy()**2)
        dist_mat_err_MG[j,i] = err
        dist_mat_MG[j,i] = dist_est.detach().cpu().numpy()
        t_test_MG[j,i] = t1-t0
        loss_test_MG[j,i] = criterion(dist_true_t**2,dist_est).item()

        t0 = time.time()
        out = net_E(input)
        t1 = time.time()
        dist_est = utils.distance_matrix(out)
        err = np.abs(dist_est.detach().cpu().numpy()-dist_true_t.detach().cpu().numpy()**2)
        dist_mat_err_E[j,i] = err
        dist_mat_E[j,i] = dist_est.detach().cpu().numpy()
        t_test_E[j,i] = t1-t0
        loss_test_E[j,i] = criterion(dist_true_t**2,dist_est).item()

        t0 = time.time()
        out = net_Hyperbolic(input)
        t1 = time.time()
        dist_est = utils.distance_hyperbolic(out)**2
        err = np.abs(dist_est.detach().cpu().numpy()-dist_true_t.detach().cpu().numpy()**2)
        dist_mat_err_Hyperbolic[j,i] = err
        dist_mat_Hyperbolic[j,i] = dist_est.detach().cpu().numpy()
        t_test_Hyperbolic[j,i] = t1-t0
        loss_test_Hyperbolic[j,i] = criterion(dist_true_t**2,dist_est).item()

        dist_mat[j,i] = dist_true_t.fill_diagonal_(1).detach().cpu().numpy()**2

#######################################
### Display loss and errors as plots
#######################################
# plt.figure(1)
# plt.clf()
# plt.plot(np.array(Ndim_list),loss_test_E.mean(0),c='r',label='Euclidean')
# plt.plot(np.array(Ndim_list),loss_test_MG.mean(0),c='b',label='MG')
# plt.legend()

distortion_E = dist_mat_E/dist_mat
distortion_Hyperbolic = dist_mat_Hyperbolic/dist_mat
distortion_MG =np.abs(dist_mat_MG)/dist_mat
for j, n_rep in enumerate(range(Nrep)):
    for i, Ndim in enumerate(Ndim_list):
        np.fill_diagonal(distortion_E[j,i],1)
        np.fill_diagonal(distortion_Hyperbolic[j,i],1)
        np.fill_diagonal(distortion_MG[j,i],1)

q = 0.99
distortion_E = (np.quantile(distortion_E,q,axis=(2,3))/np.quantile(distortion_E,1-q,axis=(2,3))).mean(0)
distortion_Hyperbolic = (np.quantile(distortion_Hyperbolic,q,axis=(2,3))/np.quantile(distortion_Hyperbolic,1-q,axis=(2,3))).mean(0)
distortion_MG = (np.quantile(distortion_MG,q,axis=(2,3))/np.quantile(distortion_MG,1-q,axis=(2,3))).mean(0)

plt.figure(2)
plt.clf()
plt.plot(np.array(Ndim_list),distortion_E,c='r',label='Euclidean')
plt.plot(np.array(Ndim_list),distortion_Hyperbolic,c='k',label='Hyperbolic')
plt.plot(np.array(Ndim_list),distortion_MG,c='b',label='MG')
plt.legend()

plt.figure(3)
plt.clf()
plt.plot(np.array(Ndim_list),np.abs(dist_mat-dist_mat_E).mean((0,2,3)),c='r',label='Est. Euclidean')
plt.plot(np.array(Ndim_list),np.abs(dist_mat-dist_mat_Hyperbolic).mean((0,2,3)),c='k',label='Est. Hyperbolic')
plt.plot(np.array(Ndim_list),np.abs(dist_mat-dist_mat_MG).mean((0,2,3)),c='b',label='Est. MG')
plt.legend()

# plt.figure(4)
# plt.clf()
# for k in range(Nrep):
#     plt.plot(np.array(Ndim_list),loss_test_E[k],c='r',label='Euclidean')
#     plt.plot(np.array(Ndim_list),loss_test_MG[k],c='b',label='MG')
# plt.legend()

np.savetxt("results/"+model_name+"/losses.txt", 
    (np.array(Ndim_list),np.abs(dist_mat-dist_mat_E).mean((0,2,3)),
    np.array(Ndim_list),np.abs(dist_mat-dist_mat_Hyperbolic).mean((0,2,3)),
    np.array(Ndim_list),np.abs(dist_mat-dist_mat_MG).mean((0,2,3))),fmt='%.3f',delimiter=',')


#######################################
### Sphere dim 3: error with icosphere and make video
#######################################
Ndim = 3
n_rep = 0
inDim = Ndim+10

if os.path.isfile("results/"+model_name_sampling+"/pts_sampled_"+str(inDim)+"_Ndim_"+str(Ndim)+".npz"):
    dat = np.load("results/"+model_name_sampling+"/pts_sampled_"+str(inDim)+"_Ndim_"+str(Ndim)+".npz")
    X_anchor = dat['sampledPts']
    inDim_ = dat['N']
    K_ = dat['K']
    tau_ = dat['tau']
    print("Load anchors")
    X_anchor_t = torch.tensor(X_anchor).type(torch_type).to(device)
    inDim = X_anchor.shape[0]

net_MG = models.MG2_transformer(inDim, outDim, N_latent=Nlatent, p=0., bn=False).to(device).train()
checkpoint = torch.load("results/"+model_name_MG+"/net_Ndim_"+str(Ndim)+"_"+str(n_rep)+".pt",map_location=device)
net_MG.load_state_dict(checkpoint['model_state_dict'])
net_MG = net_MG.eval()
net_MG = net_MG.train(False)

from icosphere import icosphere
nu = 6 # or any other integer
vertices, faces = icosphere(nu)
ax1,ax2,ax3=vertices[:,0],vertices[:,1],vertices[:,2]
X_full = np.concatenate((ax1[:,None],ax2[:,None],ax3[:,None]),1)
X_full = torch.tensor(X_full).type(torch_type).to(device)

dist_true_full = torch.acos(torch.clamp(torch.matmul(X_full,X_full.transpose(1,0)),-1+eps,1-eps))/np.pi
dist_est_full = torch.zeros_like(dist_true_full)
out_full = torch.zeros((ax1.shape[0],outDim//3,3),dtype=torch_type,device=device)

Ntest = ax1.shape[0]
with torch.no_grad():
    for k in range(Ntest//batch_size+1):
        # X = generate_points_t(batch_size)
        X = np.concatenate((ax1[k*batch_size:(k+1)*batch_size,None],ax2[k*batch_size:(k+1)*batch_size,None],ax3[k*batch_size:(k+1)*batch_size,None]),1)
        X = torch.tensor(X).type(torch_type).to(device)
        input = torch.acos(torch.clamp(torch.matmul(X,X_anchor_t.transpose(1,0)),-1+eps,1-eps))/np.pi
        dist_true_t = torch.acos(torch.clamp(torch.matmul(X,X.transpose(1,0)),-1+eps,1-eps))/np.pi
        out = net_MG(input)
        out_full[k*batch_size:(k+1)*batch_size] = out

dist_est_full = utils.dist_W2_MG_1D(out_full)
err_dist = torch.abs(dist_true_full**2-dist_est_full)
ii_, jj_ = np.where(np.eye(out_full.shape[0])==1)
err_dist[ii_,jj_] = 0
err_test_max = torch.max(err_dist,1)[0]
err_test_max = err_test_max.detach().cpu().numpy()
ind_test_max = torch.max(err_dist,1)[1]
ind_test_max = ind_test_max.detach().cpu().numpy()
err_test_mean = torch.mean(err_dist,1)
err_test_mean = err_test_mean.detach().cpu().numpy()
dist_arg_max = torch.acos(torch.clamp(torch.sum(X_full*X_full[ind_test_max],1),-1+eps,1-eps))/np.pi


fig = plt.figure(1)
plt.clf()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X_anchor[:, 0], X_anchor[:, 1], X_anchor[:, 2],marker='*',s=500,c='k')
sc = ax.scatter(X_full[:, 0].detach().cpu(), X_full[:, 1].detach().cpu(), X_full[:, 2].detach().cpu(),'o',s=err_test_max*50,c=err_test_max,cmap='jet')
plt.colorbar(sc)
plt.savefig("results/"+model_name+"/sphere_dist_max.png")
def init():
    ax.view_init(elev=10., azim=0)
    return [sc]
def animate(i):
    ax.view_init(elev=10., azim=i)
    return [sc]
# Animate
anim = animation.FuncAnimation(fig, animate, init_func=init,
                               frames=360, interval=20, blit=True)
# Save
anim.save("results/"+model_name+"/sphere_dist_max.mp4", fps=30)#, extra_args=['-vcodec', 'libx264'])

fig = plt.figure(2)
plt.clf()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X_anchor[:, 0], X_anchor[:, 1], X_anchor[:, 2],marker='*',s=500,c='k')
sc = ax.scatter(X_full[:, 0].detach().cpu(), X_full[:, 1].detach().cpu(), X_full[:, 2].detach().cpu(),'o',s=err_test_mean*500,c=err_test_mean,cmap='jet')
plt.colorbar(sc)
plt.savefig("results/"+model_name+"/sphere_dist_mean.png")

def init():
    ax.view_init(elev=10., azim=0)
    return [sc]
def animate(i):
    ax.view_init(elev=10., azim=i)
    return [sc]
# Animate
anim = animation.FuncAnimation(fig, animate, init_func=init,
                               frames=360, interval=20, blit=True)
# Save
anim.save("results/"+model_name+"/sphere_dist_mean.mp4", fps=30)#, extra_args=['-vcodec', 'libx264'])


id_ = np.argmin(np.sum((X_full.detach().cpu().numpy()-np.array([[0,1,0]]))**2,1))
fig = plt.figure(3)
plt.clf()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X_anchor[:, 0], X_anchor[:, 1], X_anchor[:, 2],marker='*',s=500,c='k')
ax.scatter(X_full[id_, 0].detach().cpu(), X_full[id_, 1].detach().cpu(), X_full[id_, 2].detach().cpu(),marker='3',s=500,c='m')
sc = ax.scatter(X_full[:, 0].detach().cpu(), X_full[:, 1].detach().cpu(), X_full[:, 2].detach().cpu(),'o',s=err_dist[id_].detach().cpu().numpy()*50,c=err_dist[id_].detach().cpu().numpy(),cmap='jet')
plt.colorbar(sc)
plt.savefig("results/"+model_name+"/sphere_dist_spatial_max.png")
def init():
    ax.view_init(elev=10., azim=0)
    return [sc]
def animate(i):
    ax.view_init(elev=10., azim=i)
    return [sc]
# Animate
anim = animation.FuncAnimation(fig, animate, init_func=init,
                               frames=360, interval=20, blit=True)
# Save
anim.save("results/"+model_name+"/sphere_dist_spatial_max.mp4", fps=30)#, extra_args=['-vcodec', 'libx264'])


#######################################
### Sphere dim 3 plot fixed continuity
### Plotly
#######################################
## Frame by frame
Nfr = 21

# Load points to deal with
Nfr = (Nfr//2)*2+1
import matplotlib as mpl
def colorFader(c1,c2,mix=0): #fade (linear interpolate) from color c1 (at mix=0) to c2 (mix=1)
    c1=np.array(mpl.colors.to_rgb(c1))
    c2=np.array(mpl.colors.to_rgb(c2))
    return mpl.colors.to_hex((1-mix)*c1 + mix*c2)

c1='#ff0000' #blue
c2='#ffe800' #green
theta_z = -np.pi/5
col_gen = lambda i: colorFader(c1,c2,np.linspace(0,1,Nfr//2+1)[i])
lin_theta = np.linspace(0,2*np.pi,Nfr,endpoint=False)
col_list1 = [col_gen(i) for i in np.concatenate((np.arange(Nfr//2),np.arange(Nfr//2,-1,-1)))]
pts_sample1 = np.array([np.cos(lin_theta)*np.cos(theta_z),np.sin(lin_theta)*np.cos(theta_z),np.ones(Nfr)*np.sin(theta_z)]).T

c1='#8fff00' #blue
c2='#00a6ff' #green
theta_z = np.pi*0
col_gen = lambda i: colorFader(c1,c2,np.linspace(0,1,Nfr//2+1)[i])
lin_theta = np.linspace(0,2*np.pi,Nfr,endpoint=False)
col_list2 = [col_gen(i) for i in np.concatenate((np.arange(Nfr//2),np.arange(Nfr//2,-1,-1)))]
pts_sample2 = np.array([np.cos(lin_theta)*np.cos(theta_z),np.sin(lin_theta)*np.cos(theta_z),np.ones(Nfr)*np.sin(theta_z)]).T

c1='#003eff' #blue
c2='#ff0080' #green
theta_z = np.pi/5
col_gen = lambda i: colorFader(c1,c2,np.linspace(0,1,Nfr//2+1)[i])
lin_theta = np.linspace(0,2*np.pi,Nfr,endpoint=False)
col_list3= [col_gen(i) for i in np.concatenate((np.arange(Nfr//2),np.arange(Nfr//2,-1,-1)))]
pts_sample3 = np.array([np.cos(lin_theta)*np.cos(theta_z),np.sin(lin_theta)*np.cos(theta_z),np.ones(Nfr)*np.sin(theta_z)]).T


col_list = col_list1+col_list2+col_list3
pts_sample = np.concatenate((pts_sample1,pts_sample2,pts_sample3),axis=0)

# Compute embedding
with torch.no_grad():
    X = torch.tensor(pts_sample).type(torch_type).to(device)
    input = torch.acos(torch.clamp(torch.matmul(X,X_anchor_t.transpose(1,0)),-1+eps,1-eps))/np.pi
    dist_true_t = torch.acos(torch.clamp(torch.matmul(X,X.transpose(1,0)),-1+eps,1-eps))/np.pi
    out = net_MG(input)

dist_true = dist_true_t.detach().cpu().numpy()
dist_est = utils.dist_W2_MG_1D(out)
err_dist = torch.abs(dist_true_t**2-dist_est)
ii_, jj_ = np.where(np.eye(out.shape[0])==1)
err_dist[ii_,jj_] = 0
err_test_max = torch.max(err_dist,1)[0]
err_test_max = err_test_max.detach().cpu().numpy()
ind_test_max = torch.max(err_dist,1)[1]
ind_test_max = ind_test_max.detach().cpu().numpy()
err_test_mean = torch.mean(err_dist,1)
err_test_mean = err_test_mean.detach().cpu().numpy()
dist_arg_max = torch.acos(torch.clamp(torch.sum(X*X[ind_test_max],1),-1+eps,1-eps))/np.pi

import plotly.graph_objects as go
import plotly.io as pio
import plotly.express as px
pio.renderers.default = "chrome"


## Make sphere3.pdf: sphere with color points
reps=0.01
u, v = np.mgrid[0:2 * np.pi:300j, 0:np.pi:300j]
x = np.cos(u) * np.sin(v)
y = np.sin(u) * np.sin(v)
z = np.cos(v)

layout = go.Layout(
    autosize=False,
    width=1000,
    height=1000
)
fig = go.Figure(data=go.Surface(x=x, y=y, z=z, colorscale=px.colors.sequential.Greys[:-4], showscale=False,
                            opacity=0.35,
                            # lighting=dict(ambient=0.5,diffuse=1,fresnel=2,specular=0.5,roughness=0.5),
                            # lightposition=dict(x=10,y=0,z=0)
                            lighting=dict(ambient=0.6, diffuse=0.8, roughness = 0.9, specular=0.2, fresnel=0.8)
                            ),
                            layout=layout)

fig.add_trace(go.Scatter3d(mode='markers',x=X_anchor[:,0]+reps*np.sign(X_anchor[:, 0]),y=X_anchor[:,1]+reps*np.sign(X_anchor[:, 1]),z=X_anchor[:,2]+reps*np.sign(X_anchor[:, 2]),
            marker_symbol='diamond', marker_color="black", marker_line_width=0, marker_size=8
            ))

fig.add_trace(go.Scatter3d(mode='markers',x=pts_sample[:,0]+reps*np.sign(pts_sample[:, 0]),
            y=pts_sample[:,1]+reps*np.sign(pts_sample[:, 1]),
            z=pts_sample[:,2]+reps*np.sign(pts_sample[:, 2]),
            marker_symbol='circle', marker_color=col_list, marker_line_width=0, marker_size=10
            ))
uu = np.linspace(0,2*np.pi,10000)
xx = np.cos(uu.reshape(-1))
yy = np.sin(uu.reshape(-1))
zz = 0*uu.reshape(-1)

fig.add_trace(go.Scatter3d(mode='markers',x=xx+reps*np.sign(xx),y=yy+reps*np.sign(yy),z=zz+reps*np.sign(zz),
            marker_symbol='circle', marker_color="black", marker_line_width=0, marker_size=0.5
            ))
fig.update_scenes(xaxis_visible=False, yaxis_visible=False,zaxis_visible=False )

camera = dict(
    up=dict(x=0, y=0, z=1),
    center=dict(x=0, y=0, z=0),
    eye=dict(x=1.5, y=1.5, z=0.5)
    )
fig.update_layout(scene_camera=camera)
fig.update_traces(showlegend=False)
fig.update_coloraxes(showscale=False)
fig.update(layout_showlegend=False)
fig.update(layout_coloraxis_showscale=False)
fig.write_image("results/"+model_name+"/sphere3.pdf")
fig.show()



## help visualize with matplolib to then generate final plotly image
## Display distribution
sp_edge = 3.5
out_np = out.detach().cpu().numpy()
mu_min = out_np[:,:,0].min()
mu_max = out_np[:,:,0].max()
sigma_min = out_np[:,:,1].min()
sigma_max = out_np[:,:,1].max()
w_min = out_np[:,:,2].min()
w_max = out_np[:,:,2].max()

from mpl_toolkits.axisartist.axislines import SubplotZero
linx = np.linspace(mu_min-sp_edge,mu_max+sp_edge,100)
y_min=0
y_max=0.8
fs = lambda x: np.log10(x+2.1)
fw = lambda x: x**0.5
plt.rcParams['xtick.labelsize']=20
plt.rcParams['ytick.labelsize']=20

fig = plt.figure(1,figsize=(15,7))
for i in range(3):
    ax = SubplotZero(fig, 131+i)
    fig.add_subplot(ax)
    plt.setp(ax, xlim=(0,linx.max()-linx.min()), ylim=(y_min, y_max))
    ax.get_xaxis().set_ticks(np.arange(1,5,1))
    ax.get_yaxis().set_ticks(ax.get_yaxis().get_ticklocs()[1:-1])
    for direction in ["xzero", "yzero"]:
        ax.axis[direction].set_axisline_style("-|>")
        ax.axis[direction].set_visible(True)

    for direction in ["left", "right", "bottom", "top"]:
        # hides borders
        ax.axis[direction].set_visible(False)

    for k in range(Nfr):
        gauss = np.zeros_like(linx)
        for j in range(out_np.shape[1]):
            gauss += fw(out_np[k+i*Nfr,j,2])*np.exp(-(linx-out_np[k+i*Nfr,j,0])**2/(2*fs(out_np[k+i*Nfr,j,1])**2))/(np.sqrt(2*np.pi)*fs(out_np[k+i*Nfr,j,1]))
        ax.plot(linx-linx.min(),gauss,c=col_list[k+i*Nfr],linewidth=2)

# plt.subplot(2,Nfr//2,k+1)
fs = lambda x: np.log10(x+1.1)
fw = lambda x: x**0.5
plt.figure(10)
plt.clf()
gauss = np.zeros_like(linx)
for j in range(out_np.shape[1]):
    plt.plot(fw(out_np[k+i*Nfr,j,2])*np.exp(-(linx-out_np[k+i*Nfr,j,0])**2/(2*fs(out_np[k+i*Nfr,j,1])**2)),linewidth=2)


#######################################
### Display distribution 3D
#######################################
## Make sphere3_distr.pdf: sphere with distributions
sp_edge = 3.5
out_np = out.detach().cpu().numpy()
mu_min = out_np[:,:,0].min()
mu_max = out_np[:,:,0].max()
sigma_min = out_np[:,:,1].min()
sigma_max = out_np[:,:,1].max()
w_min = out_np[:,:,2].min()
w_max = out_np[:,:,2].max()


from mpl_toolkits.axisartist.axislines import SubplotZero
linx = np.linspace(mu_min-sp_edge,mu_max+sp_edge,100)
sh_x = mu_min*4
fs = lambda x: np.log10(x+2.1)
fw = lambda x: x**0.5
reps=0.0
u, v = np.mgrid[0:2 * np.pi:300j, 0:np.pi:300j]
x = np.cos(u) * np.sin(v) 
y = np.sin(u) * np.sin(v) 
z = np.cos(v)

layout = go.Layout(
    autosize=False,
    width=1000,
    height=1000
)
fig = go.Figure(data=go.Surface(x=x, y=y, z=z, colorscale=px.colors.sequential.Greys[:-4], showscale=False,
                            opacity=0.35,
                            # lighting=dict(ambient=0.5,diffuse=1,fresnel=2,specular=0.5,roughness=0.5),
                            # lightposition=dict(x=10,y=0,z=0)
                            lighting=dict(ambient=0.6, diffuse=0.8, roughness = 0.9, specular=0.2, fresnel=0.8)
                            ),
                            layout=layout)
uu = np.linspace(0,2*np.pi,10000)
xx = np.cos(uu.reshape(-1))
yy = np.sin(uu.reshape(-1))
zz = 0*uu.reshape(-1)

fig.add_trace(go.Scatter3d(mode='markers',x=xx+reps*np.sign(xx),y=yy+reps*np.sign(yy),z=zz+reps*np.sign(zz),
            marker_symbol='circle', marker_color="black", marker_line_width=0, marker_size=0.5
            ))

scale_axis = 0.02
scale_amplitude = 0.4
for i in range(3):
    if i==0:
        big_shift_x = 0
        big_shift_y = 0
        big_shift_z = -0.85
        shift = np.array([[big_shift_x],[big_shift_y],[big_shift_z]])
        scale = 2.5
    if i==1:
        big_shift_x = 0
        big_shift_y = 0
        big_shift_z = -0.1
        shift = np.array([[big_shift_x],[big_shift_y],[big_shift_z]])
        scale = 3.5
    if i==2:
        big_shift_x = 0
        big_shift_y = 0
        big_shift_z = 0.65
        shift = np.array([[big_shift_x],[big_shift_y],[big_shift_z]])
        scale = 2.5

    k_list = np.roll(np.arange(Nfr),6)
    for k in range(Nfr):
        # plt.subplot(2,Nfr//2,k+1)
        gauss = np.zeros_like(linx)
        for j in range(out_np.shape[1]):
            gauss += fw(out_np[k+i*Nfr,j,2])*np.exp(-(linx-out_np[k+i*Nfr,j,0])**2/(2*fs(out_np[k+i*Nfr,j,1])**2))/(np.sqrt(2*np.pi)*fs(out_np[k+i*Nfr,j,1]))
        
        # df = np.array([np.zeros_like(linx)+k/10.,linx-linx.min(),gauss])
        th = np.pi*2.*k/Nfr
        xx = np.cos(th)*(np.zeros_like(linx)+k/10.) + np.sin(th)*(linx-linx.min()+sh_x*scale)
        yy = -np.sin(th)*(np.zeros_like(linx)+k/10.) + np.cos(th)*(linx-linx.min()+sh_x*scale)
        df = np.array([xx*scale_axis,yy*scale_axis,scale_amplitude*gauss])+shift

        fig.add_trace(go.Scatter3d(mode='lines',x=df[0],y=df[1],z=df[2],
                    marker_symbol='circle', marker_color=col_list[k_list[k]+i*Nfr], marker_line_width=10, marker_size=3,line_width=5
                    ))

fig.update_layout(showlegend=False)
fig.update_scenes(xaxis_visible=False, yaxis_visible=False,zaxis_visible=False )
camera = dict(
    up=dict(x=0, y=0, z=1),
    center=dict(x=0, y=0, z=0),
    eye=dict(x=1., y=1., z=0.5)
    )
fig.update_layout(scene_camera=camera)
fig.update_traces(showlegend=False)
fig.update_coloraxes(showscale=False)
fig.update(layout_showlegend=False)
fig.update(layout_coloraxis_showscale=False)
fig.write_image("results/"+model_name+"/sphere3_distr.pdf")
fig.show()




# for k in range(1,Nfr):
#         # plt.subplot(2,Nfr//2,k+1)
#         gauss = np.zeros_like(linx)
#         for j in range(out_np.shape[1]):
#             gauss += out_np[k+i*Nfr,j,2]*np.exp(-(linx-out_np[k+i*Nfr,j,0])**2/(2*fs(out_np[k+i*Nfr,j,1])**2))/(np.sqrt(2*np.pi)*fs(out_np[k+i*Nfr,j,1]))
# fig = px.scatter_3d(df, x='sepal_length', y='sepal_width', z='petal_width',
#               color='species')

# fig.add_trace(go.Scatter3d(mode='markers',x=xx+reps*np.sign(xx),y=yy+reps*np.sign(yy),z=zz+reps*np.sign(zz),
#             marker_symbol='circle', marker_color="black", marker_line_width=0, marker_size=0.5
#             ))


# for i in range(3):
#     # ax = plt.subplot(1,3,i+1)
#     # plt.box(False)

#     ax = SubplotZero(fig, 131+i)
#     fig.add_subplot(ax)
#     plt.setp(ax, xlim=(0,linx.max()-linx.min()), ylim=(y_min, y_max))
#     ax.get_xaxis().set_ticks(np.arange(1,5,1))
#     #ax.get_xaxis().set_tick_params(labelsize=0)
#     ax.get_yaxis().set_ticks(ax.get_yaxis().get_ticklocs()[1:-1])
#     #ax.get_yaxis().set_tick_params(labelsize=5)
#     for direction in ["xzero", "yzero"]:
#         ax.axis[direction].set_axisline_style("-|>")
#         ax.axis[direction].set_visible(True)
#         # if direction=="xzero":
#         #     ax.axis[direction].axis.set_ticklabels(np.arange(1,5,1),fontsize=20)
#         # else:
#         #     ax.axis[direction].axis.set_ticklabels(ax.get_yaxis().get_ticklocs(),fontsize=20)

#     for direction in ["left", "right", "bottom", "top"]:
#         # hides borders
#         ax.axis[direction].set_visible(False)

#     for k in range(Nfr):
#         # plt.subplot(2,Nfr//2,k+1)
#         gauss = np.zeros_like(linx)
#         for j in range(out_np.shape[1]):
#             gauss += out_np[k+i*Nfr,j,2]*np.exp(-(linx-out_np[k+i*Nfr,j,0])**2/(2*fs(out_np[k+i*Nfr,j,1])**2))/(np.sqrt(2*np.pi)*fs(out_np[k+i*Nfr,j,1]))
#         ax.plot(linx-linx.min(),gauss,c=col_list[k+i*Nfr],linewidth=2)
#     # plt.xlabel("x")
#     # plt.ylabel("$\hat{T}(x)$")
#     # ax.get_xaxis().set_ticks([])
#     # ax.get_yaxis().set_ticks([])
#     # ax[i].axis('equal')
#     # plt.grid(True,which='both')
# plt.savefig("results/"+model_name+"/sphere3_distr.pdf")





#######################################
### Sphere dim 3 continuity video
#######################################
Np=100
sc_sigma = 0.1
scx = 1.5
ymax = 0.5
id_ = np.argmin(np.sum((X_full.detach().cpu().numpy()-np.array([[0,1,0]]))**2,1))
Npts = 50
Nz = 20

z_list = np.linspace(np.pi/2-0.4*np.pi,np.pi/2+0.4*np.pi,Nz)
k_list = range(Npts)
k_list_, z_list_ = np.meshgrid(k_list,z_list)
k_list_ = k_list_.reshape(-1)
z_list_ = z_list_.reshape(-1)

# i = 0
# import gif
# @gif.frame
# def plot(i):


#     reps=0.01
#     u, v = np.mgrid[0:2 * np.pi:300j, 0:np.pi:300j]
#     x = np.cos(u) * np.sin(v)
#     y = np.sin(u) * np.sin(v)
#     z = np.cos(v)

#     from plotly.subplots import make_subplots
#     # fig = make_subplots(rows=1, cols=2)
#     fig = make_subplots(
#         rows=1, cols=2,
#         specs=[
#             [{'type': 'surface'}, {'type': 'xy'}]])

#     fig.add_trace(
#         # go.Surface(x=x, y=y, z=z, colorscale=px.colors.sequential.Greys[:-4], showscale=False,
#         #                     opacity=0.65,
#         #                     # lighting=dict(ambient=0.5,diffuse=1,fresnel=2,specular=0.5,roughness=0.5),
#         #                     # lightposition=dict(x=10,y=0,z=0)
#         #                     lighting=dict(ambient=0.6, diffuse=0.8, roughness = 0.9, specular=0.2, fresnel=0.8)
#         #                     ),
#         go.Surface(x=x, y=y, z=z, colorscale='Greys', showscale=False,
#                                 opacity=0.65,
#                                 # lighting=dict(ambient=0.5,diffuse=1,fresnel=2,specular=0.5,roughness=0.5),
#                                 # lightposition=dict(x=10,y=0,z=0)
#                                 lighting=dict(ambient=0.8, diffuse=0.4, roughness = 0.9, specular=0.1, fresnel=0.1)
#                                 ),
#         row=1, col=1)

#     fig.add_trace(go.Scatter3d(mode='markers',x=X_anchor[:,0]+reps*np.sign(X_anchor[:, 0]),y=X_anchor[:,1]+reps*np.sign(X_anchor[:, 1]),z=X_anchor[:,2]+reps*np.sign(X_anchor[:, 2]),
#                 marker_symbol='diamond', marker_color="black", marker_line_width=0, marker_size=5
#                 ),
#         row=1, col=1)

#     # uu = np.linspace(0,2*np.pi,10000)
#     # xx = np.cos(uu.reshape(-1))
#     # yy = np.sin(uu.reshape(-1))
#     # zz = 0*uu.reshape(-1)

#     # fig.add_trace(go.Scatter3d(mode='markers',x=xx+reps*np.sign(xx),y=yy+reps*np.sign(yy),z=zz+reps*np.sign(zz),
#     #             marker_symbol='circle', marker_color="black", marker_line_width=0, marker_size=1
#     #             ),
#     #     row=1, col=1)

#     zz = z_list_[i]
#     k = k_list_[i]
#     xref = np.array([[np.sin(k*2*np.pi/Npts)*np.sin(zz),np.cos(k*2*np.pi/Npts)*np.sin(zz),np.cos(zz)]])
#     id_ = np.argmin(np.sum((X_full.detach().cpu().numpy()-xref)**2,1))


#     fig.add_trace(go.Scatter3d(mode='markers',x=xref[:,0]+reps*np.sign(xref[:,0]),
#                 y=xref[:,1]+reps*np.sign(xref[:,1]),
#                 z=xref[:,2]+reps*np.sign(xref[:,2]),
#                 marker_symbol='circle', marker_color='#ff9333', marker_line_width=0, marker_size=6
#                 ),
#         row=1, col=1)

#     Xref = torch.tensor(xref).type(torch_type).to(device).view(1,-1)
#     input = torch.acos(torch.clamp(torch.matmul(Xref,X_anchor_t.transpose(1,0)),-1+eps,1-eps))/np.pi
#     out = net_MG(input)
#     out_np = out.detach().cpu().numpy()

#     # out_np = out_full.detach().cpu().numpy()
#     xmin = out_full[:,:,0].min().detach().cpu().numpy()
#     xmax = out_full[:,:,0].max().detach().cpu().numpy()
#     ymax_ = out_full[:,:,1].max().detach().cpu().numpy()
#     linx = np.linspace(xmin-ymax_/scx,xmax+ymax_/scx,Np)
#     dist_tot = np.zeros(Np)
#     # ax2.clear()
#     # ax2.set_ylim([0,ymax])
#     # ax2.set_xlim([xmin,xmax])
#     # ax2.set_xlabel("x")
#     # ax2.set_ylabel("y")
#     # ax2.grid(True)
#     # # ax2.set_title(str(k)+"  " + str(zz))
#     # ax2.plot(np.zeros(Np),np.linspace(0,ymax,Np),c='k',linestyle='--')
#     fs = lambda x: np.log10(x+1.1)
#     for j in range(out_np.shape[1]):
#         gauss = np.exp(-(linx-out_np[0,j,0])**2/(2*fs(out_np[0,j,1]**2)))/(np.sqrt(2*np.pi)*fs(out_np[0,j,1]))
#         gauss /= gauss.max()
#         gauss *= out_np[0,j,2]
#         dist_tot += gauss
#         # ax2.plot(linx,gauss,linewidth=2)
#         fig.add_trace(go.Scatter(x=linx-linx.min(), y=gauss),
#                     row=1, col=2)

#     # ax.plot(linx-linx.min(),gauss,c=col_list[k+i*Nfr],linewidth=2)

#     fig.update_scenes(xaxis_visible=False, yaxis_visible=False,zaxis_visible=False )
#     camera = dict(
#         up=dict(x=0, y=0, z=1),
#         center=dict(x=0, y=0, z=0),
#         eye=dict(x=1.25, y=1.25, z=0.5)
#         )
#     fig.update_layout(scene_camera=camera)
#     fig.update_traces(showlegend=False)
#     fig.update_coloraxes(showscale=False)
#     fig.update(layout_showlegend=False)
#     fig.update(layout_coloraxis_showscale=False)
#     # fig.write_image("results/"+model_name+"/sphere3.pdf")
#     # fig.show()
#     return fig

# # Construct list of frames
# frames = []
# from tqdm import tqdm
# for i in tqdm(range(990,z_list_.shape[0])):
#     frame = plot(i)
#     frames.append(frame)

# # Save gif from frames with a specific duration for each frame in ms
# gif.save(frames, "results/"+model_name+'/continuity.gif', duration=50)

# import cv2

# videodims = (100,100)
# fourcc = cv2.VideoWriter_fourcc(*'avc1')    
# video = cv2.VideoWriter("test.mp4",fourcc, 60,videodims)

# for i in tqdm(range(990,z_list_.shape[0])):
#     video.write(np.array(frame))



#######################################
### Sphere dim 3 - comparison grid
#######################################
## Make sphere_dist_max.pdf: sphere point color by the max error
err_dist = torch.abs(dist_true_full**2-dist_est_full)
ii_, jj_ = np.where(np.eye(out_full.shape[0])==1)
err_dist[ii_,jj_] = 0
err_test_max = torch.max(err_dist,1)[0]
err_test_max = err_test_max.detach().cpu().numpy()
ind_test_max = torch.max(err_dist,1)[1]
ind_test_max = ind_test_max.detach().cpu().numpy()
err_test_mean = torch.mean(err_dist,1)
err_test_mean = err_test_mean.detach().cpu().numpy()
dist_arg_max = torch.acos(torch.clamp(torch.sum(X_full*X_full[ind_test_max],1),-1+eps,1-eps))/np.pi
cmin = np.min([err_dist[id_].detach().cpu().numpy().min(),err_test_max.min(),err_test_mean.min()])
cmax = np.max([err_dist[id_].detach().cpu().numpy().max(),err_test_max.max(),err_test_mean.max()])

## MAX
layout = go.Layout(
    autosize=False,
    width=1000,
    height=1000
)
fig = go.Figure(data=go.Surface(x=x, y=y, z=z, colorscale=px.colors.sequential.Greys[:-4], showscale=False,
                            opacity=0.35,
                            # lighting=dict(ambient=0.5,diffuse=1,fresnel=2,specular=0.5,roughness=0.5),
                            # lightposition=dict(x=10,y=0,z=0)
                            lighting=dict(ambient=0.6, diffuse=0.8, roughness = 0.9, specular=0.2, fresnel=0.8)
                            ), layout=layout)

uu = np.linspace(0,2*np.pi,10000)
xx = np.cos(uu.reshape(-1))
yy = np.sin(uu.reshape(-1))
zz = 0*uu.reshape(-1)
fig.add_trace(go.Scatter3d(mode='markers',x=xx+reps*np.sign(xx),y=yy+reps*np.sign(yy),z=zz+reps*np.sign(zz),
            marker_symbol='circle', marker_color="black", marker_line_width=0, marker_size=0.5
            ))
fig.add_trace(go.Scatter3d(mode='markers',x=X_anchor[:,0]+reps*np.sign(X_anchor[:, 0]),y=X_anchor[:,1]+reps*np.sign(X_anchor[:, 1]),z=X_anchor[:,2]+reps*np.sign(X_anchor[:, 2]),
            marker_symbol='diamond', marker_color="black", marker_line_width=0, marker_size=8
            ))
X_full_ = X_full.detach().cpu().numpy()
fig.add_trace(go.Scatter3d(mode='markers',x=X_full_[:,0]+reps*np.sign(X_full_[:, 0]),
            y=X_full_[:,1]+reps*np.sign(X_full_[:, 1]),
            z=X_full_[:,2]+reps*np.sign(X_full_[:, 2]),
            marker_symbol='circle', marker_color=err_test_max, marker_line_width=0, marker_size=5,
            marker=dict(showscale=False, colorscale='jet', cmin=cmin, cmax=cmax)
            ))
fig.update_scenes(xaxis_visible=False, yaxis_visible=False,zaxis_visible=False )
camera = dict(
    up=dict(x=0, y=0, z=1),
    center=dict(x=0, y=0, z=0),
    eye=dict(x=1.25, y=1.25, z=0.5)
    )
fig.update_layout(scene_camera=camera)
fig.update_traces(showlegend=False)
fig.update_coloraxes(showscale=False)
fig.update(layout_showlegend=False)
fig.update(layout_coloraxis_showscale=False)
fig.write_image("results/"+model_name+"/sphere_dist_max.pdf")
fig.show()


## MEAN
## Make sphere_dist_mean.pdf: sphere point color by the mean error
layout = go.Layout(
    autosize=False,
    width=1000,
    height=1000
)
fig = go.Figure(data=go.Surface(x=x, y=y, z=z, colorscale=px.colors.sequential.Greys[:-4], showscale=False,
                            opacity=0.35,
                            # lighting=dict(ambient=0.5,diffuse=1,fresnel=2,specular=0.5,roughness=0.5),
                            # lightposition=dict(x=10,y=0,z=0)
                            lighting=dict(ambient=0.6, diffuse=0.8, roughness = 0.9, specular=0.2, fresnel=0.8)
                            ), layout=layout)

uu = np.linspace(0,2*np.pi,10000)
xx = np.cos(uu.reshape(-1))
yy = np.sin(uu.reshape(-1))
zz = 0*uu.reshape(-1)
fig.add_trace(go.Scatter3d(mode='markers',x=xx+reps*np.sign(xx),y=yy+reps*np.sign(yy),z=zz+reps*np.sign(zz),
            marker_symbol='circle', marker_color="black", marker_line_width=0, marker_size=0.5
            ))
fig.add_trace(go.Scatter3d(mode='markers',x=X_anchor[:,0]+reps*np.sign(X_anchor[:, 0]),y=X_anchor[:,1]+reps*np.sign(X_anchor[:, 1]),z=X_anchor[:,2]+reps*np.sign(X_anchor[:, 2]),
            marker_symbol='diamond', marker_color="black", marker_line_width=0, marker_size=8
            ))
X_full_ = X_full.detach().cpu().numpy()
fig.add_trace(go.Scatter3d(mode='markers',x=X_full_[:,0]+reps*np.sign(X_full_[:, 0]),
            y=X_full_[:,1]+reps*np.sign(X_full_[:, 1]),
            z=X_full_[:,2]+reps*np.sign(X_full_[:, 2]),
            marker_symbol='circle', marker_color=err_test_mean, marker_line_width=0, marker_size=5,
            marker=dict(showscale=False, colorscale='jet', cmin=cmin, cmax=cmax)
            ))
fig.update_scenes(xaxis_visible=False, yaxis_visible=False,zaxis_visible=False )
camera = dict(
    up=dict(x=0, y=0, z=1),
    center=dict(x=0, y=0, z=0),
    eye=dict(x=1.25, y=1.25, z=0.5)
    )
fig.update_layout(scene_camera=camera)
fig.update_traces(showlegend=False)
fig.update_coloraxes(showscale=False)
fig.update(layout_showlegend=False)
fig.update(layout_coloraxis_showscale=False)
fig.write_image("results/"+model_name+"/sphere_dist_mean.pdf")
fig.show()


## MAX from one point
## Make sphere_dist_spatial_max.pdf: sphere point color by the error to a fixed point
id_ = np.argmin(np.sum((X_full.detach().cpu().numpy()-np.array([[0,-1,0]]))**2,1))
layout = go.Layout(
    autosize=False,
    width=1000,
    height=1000
)
fig = go.Figure(data=go.Surface(x=x, y=y, z=z, colorscale=px.colors.sequential.Greys[:-4], showscale=False,
                            opacity=0.35,
                            # lighting=dict(ambient=0.5,diffuse=1,fresnel=2,specular=0.5,roughness=0.5),
                            # lightposition=dict(x=10,y=0,z=0)
                            lighting=dict(ambient=0.6, diffuse=0.8, roughness = 0.9, specular=0.2, fresnel=0.8),
                            ), layout=layout)

uu = np.linspace(0,2*np.pi,10000)
xx = np.cos(uu.reshape(-1))
yy = np.sin(uu.reshape(-1))
zz = 0*uu.reshape(-1)
fig.add_trace(go.Scatter3d(mode='markers',x=xx+reps*np.sign(xx),y=yy+reps*np.sign(yy),z=zz+reps*np.sign(zz),
            marker_symbol='circle', marker_color="black", marker_line_width=0, marker_size=0.5
            ))
fig.add_trace(go.Scatter3d(mode='markers',x=X_anchor[:,0]+reps*np.sign(X_anchor[:, 0]),y=X_anchor[:,1]+reps*np.sign(X_anchor[:, 1]),z=X_anchor[:,2]+reps*np.sign(X_anchor[:, 2]),
            marker_symbol='diamond', marker_color="black", marker_line_width=0, marker_size=8
            ))
X_full_ = X_full.detach().cpu().numpy()
fig.add_trace(go.Scatter3d(mode='markers',x=X_full_[:,0]+reps*np.sign(X_full_[:, 0]),
            y=X_full_[:,1]+reps*np.sign(X_full_[:, 1]),
            z=X_full_[:,2]+reps*np.sign(X_full_[:, 2]),
            marker_symbol='circle', marker_color=err_dist[id_].detach().cpu().numpy(), marker_line_width=0, marker_size=5,
            marker=dict(showscale=True, colorscale='jet', cmin=cmin, cmax=cmax,
                            colorbar=dict(thickness=40,len=0.7,
                                #ticklen=3, tickcolor='black',
                                tickfont=dict(size=30, color='black'))
            # colorbar=dict(lenmode='fraction', len=0.75, tickness=20)),
            )))
fig.add_trace(go.Scatter3d(mode='markers',x=X_full[id_:id_+1, 0].detach().cpu()+reps*np.sign(X_full[id_:id_+1, 0].detach().cpu()),
            y=X_full[id_:id_+1, 1].detach().cpu()+reps*np.sign(X_full[id_:id_+1, 1].detach().cpu()),
            z=X_full[id_:id_+1, 2].detach().cpu()+reps*np.sign(X_full[id_:id_+1, 2].detach().cpu()),
            marker_symbol='x', marker_color='red', marker_line_width=0, marker_size=10,
            marker=dict(showscale=False, colorscale='jet', cmin=cmin, cmax=cmax)
            ))
fig.update_scenes(xaxis_visible=False, yaxis_visible=False,zaxis_visible=False )
camera = dict(
    up=dict(x=0, y=0, z=1),
    center=dict(x=0, y=0, z=0),
    eye=dict(x=1.25, y=1.25, z=0.5)
    )
fig.update_layout(scene_camera=camera)
fig.update_traces(showlegend=False)
fig.update_coloraxes(showscale=False)
fig.update(layout_showlegend=False)
fig.update(layout_coloraxis_showscale=False)
fig.write_image("results/"+model_name+"/sphere_dist_spatial_max.pdf")
fig.show()

