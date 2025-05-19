import pandas as pd
import torch.nn.functional as F
from mat4py import loadmat
import copy
from models import NeuralNet
import random

import os
import pickle as pkl

import torch
import torch.nn as nn

import numpy as np
from torch.utils.data import Dataset, DataLoader

import matplotlib.pyplot as plt


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class GaussianNoise(nn.Module):
    def __init__(self, stddev):
        super().__init__()
        self.stddev = stddev

    def forward(self, din):
        if self.training:
            return din + torch.autograd.Variable(torch.randn(din.size()) * self.stddev)
        return din

class Fsmm(nn.Module):
    def __init__(self,input_size,hidden_size,gaussian_noise = 0.0):
        super(Fsmm, self).__init__()
        self.mlp = NeuralNet(input_size,hidden_size,1)
        self.Q = nn.Parameter(torch.tensor(2.9),requires_grad=True)
        self.delta = nn.Parameter(torch.tensor(1.0),requires_grad=True)
        self.noise = GaussianNoise(gaussian_noise)
        self.f_f_soc1 = nn.Linear(1, hidden_size)
        self.f_f_soc2 = nn.Linear(hidden_size, 1)
        self.U0 = nn.Parameter(torch.tensor(1.0),requires_grad=True)
        self.R0 = nn.Parameter(torch.tensor(1.0),requires_grad=True)
        self.C1 = nn.Parameter(torch.tensor(1.0),requires_grad=True)
        self.R1 = nn.Parameter(torch.tensor(1.0),requires_grad=True)

    def forward(self,x):
        out = x #V I T time
        batch_size,a = x.shape
        out = torch.autograd.Variable(out,requires_grad = True)
        soc = self.mlp(out)#mlp
        weight = torch.ones_like(soc)
        lamada_i = self.delta/self.Q * out[:,1:2] #电流的函数
        d_soc = torch.autograd.grad(soc,out,grad_outputs=weight,retain_graph=True,allow_unused=True,create_graph=True)[0][:,-1]
        d_soc = d_soc.reshape(batch_size,-1)
        loss1 = lamada_i + d_soc#第一个式子
        f_soc = self.f_f_soc2(self.f_f_soc1(soc))
        u1 = out[:,0:1] - f_soc - out[:,1:2] * self.R0
        weight = torch.ones_like(u1)
        d_u1 = torch.autograd.grad(u1,out,grad_outputs=weight,retain_graph=True,allow_unused=True,create_graph=True)[0][:,-1]
        loss2 = d_u1 + self.delta / (self.R1*self.C1) * u1 - out[:,1:2] / self.C1  #第二个方程
        return soc,loss1,loss2

class MatDNNDataset(Dataset):
    def __init__(self, root):
        self.data = loadmat(root)

        # Panasonic Ah capacity
        self.BATTERY_AH_CAPACITY = 2.9000

        # Construct dataframe from MATLAB data
        self.df = pd.DataFrame(self.data)
        self.df = self.df.T
        self.df = self.df.apply(lambda x : pd.Series(x[0]))
        self.df = self.df.applymap(lambda x : x[0])

        # Clean up unnecessary columns
        del self.df['Chamber_Temp_degC']
        del self.df['TimeStamp']
        del self.df['Power']
        del self.df['Wh']

        # Add SOC column
        ah = self.df['Ah']
        self.df['SOC'] = 1 + (ah/self.BATTERY_AH_CAPACITY)
        del self.df['Ah']

        # Convert data to numpy
        self.V = self.df['Voltage'].to_numpy(dtype=np.float32)
        self.I = self.df['Current'].to_numpy(dtype=np.float32)
        self.T = self.df['Battery_Temp_degC'].to_numpy(dtype=np.float32)
        self.soc = self.df['SOC'].to_numpy(dtype=np.float32)
        self.time = self.df['Time'].to_numpy(dtype=np.float32)
        length = len(self.soc)

        # Set values for dataset
        self.V = torch.from_numpy(self.V)
        self.I = torch.from_numpy(self.I)
        self.T = torch.from_numpy(self.T)
        self.y = torch.from_numpy(self.soc)
        self.time = torch.from_numpy(self.time)
        # Reshape to match required label tensor shape
        self.y = self.y.reshape((length, 1))
        self.V = self.V.reshape((length, 1))
        self.I = self.I.reshape((length, 1))
        self.T = self.T.reshape((length, 1))
        self.time = self.time.reshape((length,1))
        self.X = torch.cat((self.V,self.I,self.T,self.time),1)

    def __getitem__(self, idx):
        return self.X[idx, :],self.y[idx]

    def __len__(self):
        return len(self.y)

"""  0 deg"""
input_size = 4
hidden_size = 32
model = Fsmm(input_size,hidden_size).to(device)
model_mlp = NeuralNet(input_size,hidden_size,1).to(device)
model = torch.load('pinn_change_new_saves/model_pinn.pt').to(device)
model_mlp = torch.load('pinn_change_new_saves/model_mlp.pt').to(device)
save_dir = 'pinn_change_new_saves'
save_dir = os.path.join(save_dir)
validation_dataset = MatDNNDataset("data/Panasonic 18650PF Data/0degC/Drive cycles/06-02-17_10.43 0degC_HWFET_Pan18650PF.mat")

a = list()
soc = list()
y_fsmm = list()
loss_fsmm = list()
y_mlp = list()
loss_mlp = list()
for i,u in enumerate(validation_dataset):
    x,y = u[0].to(device),u[1].to(device)
    soc.append(y.item())
    x = x.reshape(-1,4)
    a.append(x[0,-1].item())
    y_pre,_,_ = model(x)
    y_fsmm.append(y_pre.item())
    loss_fsmm.append(y_pre.item() - y.item())
    y_pre = model_mlp(x)
    loss_mlp.append(y_pre.item() - y.item())
    y_mlp.append(y_pre.item())

y_fsmm,y_mlp = np.array(y_fsmm),np.array(y_mlp)
fig,axes = plt.subplots(1,1,figsize=(28,20))
axes.plot(a,soc,color='b',label='Acture SOC')
axes.plot(a,y_fsmm,color='g',label='MLP')
axes.plot(a,y_mlp,color='r',label='MLP+PINN')
axes.legend()
plt.ylabel('SOC')
plt.xlabel('Time (s) T=0°C')
plt.savefig(os.path.join(save_dir, 'soc_pred.jpg'), bbox_inches='tight', pad_inches=0)



loss_fsmm,loss_mlp = np.array(loss_fsmm),np.array(loss_mlp)
fig,axes = plt.subplots(1,1,figsize=(28,20))
axes.plot(a,loss_fsmm,color='g',label='MLP')
axes.plot(a,loss_mlp,color='r',label='MLP+PINN')
axes.legend()
plt.ylabel('SOC error')
plt.xlabel('Time (s) T=0°C')

plt.savefig(os.path.join(save_dir, 'soc_error.jpg'), bbox_inches='tight', pad_inches=0)