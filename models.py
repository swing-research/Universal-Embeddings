import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_model_summary import summary



"""
Implentation of the proposed Probabilistic transformer described in the paper.

INPUT:
 -inDim: number of anchors.
 -outDim: number of mixtures. 
 -N_latent: dimension of the latent layers.
 -weights: if False, only two parameters are estimated for each components.
 -p: dropout parameter, 0 is none.
 -bn: boolean specifying if use of batch normalization.
"""
class MG2_transformer(nn.Module):
    def __init__(self,inDim, outDim, N_latent=16, weights=True, p=0., bn=True):
        super(MG2_transformer, self).__init__()
        self.inDim = inDim
        self.outDim = outDim
        self.weights = weights
        self.bn = bn
        self.p = p

        self.N_latent= N_latent
        self.Gauss_lin1 = nn.Linear(self.inDim, self.N_latent)
        self.Gauss_lin2 = nn.Linear(self.N_latent, self.N_latent)
        self.Gauss_lin3 = nn.Linear(self.N_latent, self.N_latent)
        self.Gauss_lin4 = nn.Linear(self.N_latent, self.N_latent)
        self.Gauss_lin5 = nn.Linear(self.N_latent, self.N_latent)
        self.Gauss_lin6 = nn.Linear(self.N_latent*5, self.outDim)

        if self.bn:
            self.bn1 = nn.BatchNorm1d(self.N_latent)
            self.bn2 = nn.BatchNorm1d(self.N_latent)
            self.bn3 = nn.BatchNorm1d(self.N_latent)
            self.bn4 = nn.BatchNorm1d(self.N_latent)
            self.bn5 = nn.BatchNorm1d(self.N_latent)

    def Gauss_basis(self,x):
        y1 = F.relu(self.Gauss_lin1(x))
        if self.p!=0:
            y1 = nn.functional.dropout(y1, p=self.p, training=self.training)
        if self.bn:
            y1 = self.bn1(y1)
        y2 = F.relu(self.Gauss_lin2(y1))
        if self.p!=0:
            y2 = nn.functional.dropout(y2, p=self.p, training=self.training)
        if self.bn:
            y2 = self.bn2(y2)
        y3 = F.relu(self.Gauss_lin3(y2))
        if self.p!=0:
            y3 = nn.functional.dropout(y3, p=self.p, training=self.training)
        if self.bn:
            y3 = self.bn3(y3)
        y4 = F.relu(self.Gauss_lin4(y3))
        if self.p!=0:
            y4 = nn.functional.dropout(y4, p=self.p, training=self.training)
        if self.bn:
            y4 = self.bn4(y4)
        y5 = F.relu(self.Gauss_lin5(y4))
        if self.p!=0:
            y5 = nn.functional.dropout(y5, p=self.p, training=self.training)
        if self.bn:
            y5 = self.bn5(y5)
        y6 = torch.cat((y1,y2,y3,y4,y5),dim=1)
        y = self.Gauss_lin6(y6)
        if self.weights:
            return y.view(-1,self.outDim//3,3)
        else:
            return y.view(-1,self.outDim//2,2)

    def forward(self,x):
        out = self.Gauss_basis(x)
        if self.weights:
            out = torch.cat((out[:,:,0:1],torch.abs(out[:,:,1:2]),nn.Softmax(dim=1)(out[:,:,2:3])),2)
        else:
            out = torch.cat((out[:,:,0:1],torch.abs(out[:,:,1:2])),2)
        if self.outDim==2:
            out = out.view(-1,self.outDim)
        return out
    
    def summary(self):
        print("Network parameters")
        if next(self.parameters()).is_cuda:
            print(summary(self, torch.zeros((1,self.inDim)).cuda(), show_input=True))
        else:
            print(summary(self, torch.zeros((1,self.inDim)), show_input=True))
        print("Number of learnable parameters: {0}".format(sum(p.numel() for p in self.parameters() if p.requires_grad)))


"""
Multilayer perceptron simple form.

Maps signal of size (batch,inDim) to signal of size (batch,outDim).

INPUTS:
 -outDim: number of channels in the last layer. 
 -N_latent: width of the hidden layers.
"""
class NetMLP(nn.Module):
    def __init__(self,inDim, outDim, N_latent=8, p=0., bn=True, hyperbolic=False):
        super(NetMLP, self).__init__()
        self.N_latent = N_latent
        self.inDim = inDim
        self.outDim = outDim
        self.p = p
        self.bn = bn
        self.hyperbolic = hyperbolic

        self.Gauss_lin1 = nn.Linear(self.inDim, self.N_latent)
        self.Gauss_lin2 = nn.Linear(self.N_latent, self.N_latent)
        self.Gauss_lin3 = nn.Linear(self.N_latent, self.N_latent)
        self.Gauss_lin4 = nn.Linear(self.N_latent, self.N_latent)
        self.Gauss_lin5 = nn.Linear(self.N_latent, self.N_latent)
        self.Gauss_lin6 = nn.Linear(self.N_latent*5, self.outDim)

        if self.bn:
            self.bn1 = nn.BatchNorm1d(self.N_latent)
            self.bn2 = nn.BatchNorm1d(self.N_latent)
            self.bn3 = nn.BatchNorm1d(self.N_latent)
            self.bn4 = nn.BatchNorm1d(self.N_latent)
            self.bn5 = nn.BatchNorm1d(self.N_latent)

    def forward(self,x):
        y1 = F.relu(self.Gauss_lin1(x))
        if self.p!=0:
            y1 = nn.functional.dropout(y1, p=self.p, training=self.training)
        if self.bn:
            y1 = self.bn1(y1)
        y2 = F.relu(self.Gauss_lin2(y1))
        if self.p!=0:
            y2 = nn.functional.dropout(y2, p=self.p, training=self.training)
        if self.bn:
            y2 = self.bn2(y2)
        y3 = F.relu(self.Gauss_lin3(y2))
        if self.p!=0:
            y3 = nn.functional.dropout(y3, p=self.p, training=self.training)
        if self.bn:
            y3 = self.bn3(y3)
        y4 = F.relu(self.Gauss_lin4(y3))
        if self.p!=0:
            y4 = nn.functional.dropout(y4, p=self.p, training=self.training)
        if self.bn:
            y4 = self.bn4(y4)
        y5 = F.relu(self.Gauss_lin5(y4))
        if self.p!=0:
            y5 = nn.functional.dropout(y5, p=self.p, training=self.training)
        if self.bn:
            y5 = self.bn5(y5)
        y6 = torch.cat((y1,y2,y3,y4,y5),dim=1)
        y = self.Gauss_lin6(y6)
        if self.hyperbolic:
            y = torch.cat((y[:,:-1],torch.abs(y[:,self.outDim-2:-1])),1)
        return y
    
    def summary(self):
        print("MLP network:")
        if next(self.parameters()).is_cuda:
            print(summary(self, torch.zeros((1,self.inDim)).cuda(), show_input=True))
        else:
            print(summary(self, torch.zeros((1,self.inDim)), show_input=True))









"""
Transformer network as in Theorem 3.1.

Maps signal of size (batch,1) to signal of size (batch,4).

INPUTS:
 -N_latent: width of the hidden layers.
"""
class NetTransformer(nn.Module):
    def __init__(self,outDim, N_latent=8):
        super(NetTransformer, self).__init__()
        self.N_latent = N_latent
        self.outDim = outDim
        self.lin1 = nn.Linear(1, self.N_latent)
        self.lin2 = nn.Linear(self.N_latent, self.N_latent)
        self.lin3 = nn.Linear(self.N_latent, self.N_latent)
        self.lin4 = nn.Linear(self.N_latent, self.outDim)

    def forward(self, x):
        x = F.relu(self.lin1(x))
        x = F.relu(self.lin2(x))
        x = F.relu(self.lin3(x))
        x = self.lin4(x)
        x = nn.Softmax(dim=1)(x)
        return x
    
    def summary(self):
        print("Transformer network:")
        if next(self.parameters()).is_cuda:
            print(summary(self, torch.zeros((1,1)).cuda(), show_input=True))
        else:
            print(summary(self, torch.zeros((1,1)), show_input=True))

# net_transf = NetTransformer(outDim=2,N_latent=20*Npts).to(device).train()
# net_transf.summary()

# net_transf_tot = []
# params = []
# for k in range(Npts):
#     net_transf_tot.append(NetTransformer(outDim=2,N_latent=20*Npts).to(device).train())
#     params += list(net_transf_tot[k].parameters())
# T = lambda x: net_transf(x)

"""
Full generator of Gaussian mixture.
Input is process by a Transformer to get weights with some Gaussian distributions
where mean and variance can be learned.
"""
class NetGauss1d(nn.Module):
    def __init__(self,NetTransformer, outDim=4):
        super(NetGauss1d, self).__init__()
        self.outDim = outDim
        self.NetTransformer = NetTransformer
        self.Gauss_basis = nn.Parameter(20*(torch.rand(self.outDim,2)-0.5), requires_grad=True)

    def forward(self, x):
        w = self.NetTransformer(x)
        Tx = torch.matmul(w,self.Gauss_basis)

        out = torch.zeros_like(Tx)
        out[:,0] = Tx[:,0]
        out[:,1] = torch.exp(Tx[:,1])

        return out
    
    def summary(self):
        print("Full network:")
        if next(self.parameters()).is_cuda:
            print(summary(self, torch.zeros((1,1)).cuda(), show_input=True))
        else:
            print(summary(self, torch.zeros((1,1)), show_input=True))

        print("Number of learnable parameters: {0}".format(sum(p.numel() for p in self.parameters() if p.requires_grad)))


# net_transf = NetTransformer(outDim=4,N_latent=2*Npts).to(device).train()
# net_full = NetGauss1d(net_transf,outDim=4).to(device).train()
# net_full.summary()


# # Fixed for now, could be learn?
# Gauss_basis = np.zeros((4,2))
# Gauss_basis[:,0] = np.linspace(0,1,4)
# Gauss_basis[:,1] = 1
# Gauss_basis_t = torch.tensor(Gauss_basis).type(torch_type).to(device)
# T = lambda x: torch.matmul(net_transf(x),Gauss_basis_t)


"""
Full generator of Gaussian mixture.
Input is process by a Transformer to get weights with some Gaussian distributions
where mean and variance can be learned using  MLP taking index as input.
Trnasformer network is defined inside this architecture.
"""
class NetGauss1dMLP(nn.Module):
    def __init__(self,inDim=1, outDim=4, N_latent1=20, N_latent2=20):
        super(NetGauss1dMLP, self).__init__()
        self.inDim = inDim
        self.outDim = outDim

        self.N_latent1 = N_latent1
        self.N_latent2 = N_latent2
        self.Gauss_lin1 = nn.Linear(self.inDim, self.N_latent2)
        self.Gauss_lin2 = nn.Linear(self.N_latent2, self.N_latent2)
        self.Gauss_lin3 = nn.Linear(self.N_latent2, self.N_latent2)
        self.Gauss_lin4 = nn.Linear(self.N_latent2*3, self.outDim*2)

        self.lin1 = nn.Linear(self.inDim, self.N_latent1)
        self.lin2 = nn.Linear(self.N_latent1, self.N_latent1)
        self.lin3 = nn.Linear(self.N_latent1, self.N_latent1)
        self.lin4 = nn.Linear(self.N_latent1*3, self.outDim)

    def Transformer(self, x):
        x1 = F.relu(self.lin1(x))
        x2 = F.relu(self.lin2(x1))
        x3 = F.relu(self.lin3(x2))
        x4 = torch.cat((x1,x2,x3),dim=1)
        x = self.lin4(x4)
        x = nn.Softmax(dim=1)(x)
        return x

    def Gauss_basis(self,x):
        y1 = self.Gauss_lin1(x)
        y2 = self.Gauss_lin2(y1)
        y3 = self.Gauss_lin3(y2)
        y4 = torch.cat((y1,y2,y3),dim=1)
        y = self.Gauss_lin4(y4).view(-1,self.outDim,2)
        return y

    def forward(self, x):
        w = self.Transformer(x)
        y = self.Gauss_basis(x)
        y = torch.cat((y[:,:,0:1],torch.exp(y[:,:,1:2])),2)
        out = torch.matmul(w.view(-1,1,w.shape[1]),y).view(-1,2)
        # import ipdb; ipdb.set_trace()

        # out = torch.zeros_like(Tx)
        # out[:,0] = Tx[:,0]
        # out[:,1] = torch.exp(Tx[:,1])

        return out
    
    def summary(self):
        print("Full with MLP for Gauss basis network:")
        if next(self.parameters()).is_cuda:
            print(summary(self, torch.zeros((1,self.inDim)).cuda(), show_input=True))
        else:
            print(summary(self, torch.zeros((1,self.inDim)), show_input=True))

        print("Number of learnable parameters: {0}".format(sum(p.numel() for p in self.parameters() if p.requires_grad)))



"""
Transformer to compute weights in front of the fixed Gaussian distributions.

If N_latent2=-1, then the same transformer is used to average the means and the variances.

means, variances: dim (1,outDim,1)
"""
class NetTransformerFixedGauss(nn.Module):
    def __init__(self,means, variances, inDim=1, outDim=4, N_latent1=20, N_latent2=-1):
        super(NetTransformerFixedGauss, self).__init__()
        self.inDim = inDim
        self.outDim = outDim

        self.means = means
        self.variances = variances
        self.N_latent1 = N_latent1
        self.N_latent2 = N_latent2
        if self.N_latent2==-1:
            self.Ntransformer = 1
        else:
            self.Ntransformer = 2

        self.scalings = nn.Parameter(torch.rand(1,2), requires_grad=True)

        self.lin1_1 = nn.Linear(self.inDim, self.N_latent1)
        self.lin1_2 = nn.Linear(self.N_latent1, self.N_latent1)
        self.lin1_3 = nn.Linear(self.N_latent1, self.N_latent1)
        self.lin1_4 = nn.Linear(self.N_latent1*3, self.outDim)

        if self.Ntransformer==2:
            self.lin2_1 = nn.Linear(self.inDim, self.N_latent2)
            self.lin2_2 = nn.Linear(self.N_latent2, self.N_latent2)
            self.lin2_3 = nn.Linear(self.N_latent2, self.N_latent2)
            self.lin2_4 = nn.Linear(self.N_latent2*3, self.outDim)

    def Transformer1(self, x):
        # import ipdb; ipdb.set_trace()
        x1 = F.relu(self.lin1_1(x))
        x2 = F.relu(self.lin1_2(x1))
        x3 = F.relu(self.lin1_3(x2))
        x4 = torch.cat((x1,x2,x3),dim=1)
        x = self.lin1_4(x4)
        x = nn.Softmax(dim=1)(x)
        return x

    def Transformer2(self, x):
        x1 = F.relu(self.lin2_1(x))
        x2 = F.relu(self.lin2_2(x1))
        x3 = F.relu(self.lin2_3(x2))
        x4 = torch.cat((x1,x2,x3),dim=1)
        x = self.lin2_4(x4)
        x = nn.Softmax(dim=1)(x)
        return x

    def forward(self, x):
        w1 = self.Transformer1(x)
        mean_est = torch.matmul(w1.view(-1,1,w1.shape[1]),self.means).view(-1,1)
        if self.Ntransformer==2:
            w2 = self.Transformer2(x)
            var_est = torch.matmul(w2.view(-1,1,w2.shape[1]),self.variances).view(-1,1)
        else:
            var_est = torch.matmul(w1.view(-1,1,w1.shape[1]),self.variances).view(-1,1)

        out = torch.cat((self.scalings[0,0]*mean_est,torch.exp(self.scalings[0,1]*var_est)),1)
        return out
    
    def summary(self):
        print("Network")
        if next(self.parameters()).is_cuda:
            print(summary(self, torch.zeros((1,self.inDim)).cuda(), show_input=True))
        else:
            print(summary(self, torch.zeros((1,self.inDim)), show_input=True))

        print("Number of learnable parameters: {0}".format(sum(p.numel() for p in self.parameters() if p.requires_grad)))



"""
Pytorch model to directly optimize mean and variance of each points.
"""
class DirectOptimization1D(nn.Module):
    def __init__(self,batch_size):
        super(DirectOptimization1D, self).__init__()
        self.batch_size = batch_size

        self.means = nn.Parameter(torch.rand(self.batch_size,1), requires_grad=True)
        self.variances = nn.Parameter(torch.rand(self.batch_size,1), requires_grad=True)

    def forward(self):
        out = torch.cat((self.means,torch.exp(self.variances)),1)
        return out
    
    def summary(self):
        print("Network parameters")
        if next(self.parameters()).is_cuda:
            print(summary(self, show_input=True))
        else:
            print(summary(self, show_input=True))

        print("Number of learnable parameters: {0}".format(sum(p.numel() for p in self.parameters() if p.requires_grad)))


"""
NN model to directly estimate mean and variance of each points based on some input.
"""
class DirectOptimization1D_NN(nn.Module):
    def __init__(self,inDim, outDim, N_latent=16):
        super(DirectOptimization1D_NN, self).__init__()
        self.inDim = inDim
        self.outDim = outDim

        self.N_latent= N_latent
        self.Gauss_lin1 = nn.Linear(self.inDim, self.N_latent)
        self.Gauss_lin2 = nn.Linear(self.N_latent, self.N_latent)
        self.Gauss_lin3 = nn.Linear(self.N_latent, self.N_latent)
        self.Gauss_lin4 = nn.Linear(self.N_latent, self.outDim)

    def Gauss_basis(self,x):
        y = self.Gauss_lin1(x)
        y = self.Gauss_lin2(y)
        y = self.Gauss_lin3(y)
        y = self.Gauss_lin4(y)
        return y.view(-1,self.outDim//2,2)

    def forward(self,x):
        out = self.Gauss_basis(x)
        out = torch.cat((out[:,:,0:1],torch.exp(out[:,:,1:2])),2)
        # out = torch.mean(out,0,keepdim=True)
        if self.outDim==2:
            out = out.view(-1,2)
        return out
    
    def summary(self):
        print("Network parameters")
        if next(self.parameters()).is_cuda:
            print(summary(self, torch.zeros((1,self.inDim)).cuda(), show_input=True))
        else:
            print(summary(self, torch.zeros((1,self.inDim)), show_input=True))

        print("Number of learnable parameters: {0}".format(sum(p.numel() for p in self.parameters() if p.requires_grad)))



"""
weights: if True, learn weighted Gaussian mixture, output is of size (Nmixture,3), where last element is the weight of each mixture
"""
class DirectOptimization1D_NN_v2(nn.Module):
    def __init__(self,inDim, outDim, N_latent=16, weights=False, p=0.2):
        super(DirectOptimization1D_NN_v2, self).__init__()
        self.inDim = inDim
        self.outDim = outDim
        self.weights = weights
        self.p = p

        self.N_latent= N_latent
        self.Gauss_lin1 = nn.Linear(self.inDim, self.N_latent)
        self.Gauss_lin2 = nn.Linear(self.N_latent, self.N_latent)
        self.Gauss_lin3 = nn.Linear(self.N_latent, self.N_latent)
        self.Gauss_lin4 = nn.Linear(self.N_latent, self.N_latent)
        self.Gauss_lin5 = nn.Linear(self.N_latent*4, self.outDim)

    def Gauss_basis(self,x):
        y1 = self.Gauss_lin1(x)
        y1 = nn.Dropout(p=self.p)(y1)
        y2 = self.Gauss_lin2(y1)
        y2 = nn.Dropout(p=self.p)(y2)
        y3 = self.Gauss_lin3(y2)
        y3 = nn.Dropout(p=self.p)(y3)
        y4 = self.Gauss_lin4(y3)
        y4 = nn.Dropout(p=self.p)(y4)
        y5 = torch.cat((y1,y2,y3,y4),dim=1)
        y = self.Gauss_lin5(y5)
        if self.weights:
            return y.view(-1,self.outDim//3,3)
        else:
            return y.view(-1,self.outDim//2,2)

    def forward(self,x):
        out = self.Gauss_basis(x)
        if self.weights:
            out = torch.cat((out[:,:,0:1],torch.abs(out[:,:,1:2]),nn.Softmax(dim=1)(out[:,:,2:3])),2)
        else:
            out = torch.cat((out[:,:,0:1],torch.abs(out[:,:,1:2])),2)
        # out = torch.mean(out,0,keepdim=True)
        if self.outDim==2 or self.outDim==3:
            out = out.view(-1,self.outDim)
        return out
    
    def summary(self):
        print("Network parameters")
        if next(self.parameters()).is_cuda:
            print(summary(self, torch.zeros((1,self.inDim)).cuda(), show_input=True))
        else:
            print(summary(self, torch.zeros((1,self.inDim)), show_input=True))

        print("Number of learnable parameters: {0}".format(sum(p.numel() for p in self.parameters() if p.requires_grad)))
