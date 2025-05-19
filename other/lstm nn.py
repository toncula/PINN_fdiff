import pandas as pd
import torch.nn.functional as F
from mat4py import loadmat
import copy
import random

import os
import pickle as pkl
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
import matplotlib as mpl
mpl.use('TkAgg')
import pickle
import matplotlib.pyplot as plt

import  time
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def setup_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True

class GaussianNoise(nn.Module):
    def __init__(self, stddev):
        super().__init__()
        self.stddev = stddev

    def forward(self, din):
        if self.training:
            return din + torch.autograd.Variable(torch.randn(din.size()) * self.stddev)
        return din


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, gaussian_noise):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)
        self.noise = GaussianNoise(gaussian_noise)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)

        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_size)

        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])
        return out



class Fsmm(nn.Module):
    def __init__(self,input_size,hidden_size,num_layers,gaussian_noise = 0.0):
        super(Fsmm, self).__init__()
        self.lstm = LSTM(input_size,hidden_size,num_layers,1,gaussian_noise)
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
        batch_size,seq,dim = x.shape
        out = torch.autograd.Variable(out,requires_grad = True)
        soc = self.lstm(out)#lstm
        weight = torch.ones_like(soc)
        lamada_i = self.delta/self.Q * out[:,-1,1:2] #电流的函数
        d_soc = torch.autograd.grad(soc,out,grad_outputs=weight,retain_graph=True,allow_unused=True,create_graph=True)[0][:,-1,-1]
        d_soc = d_soc.reshape(lamada_i.shape)
        loss1 = lamada_i + d_soc#第一个式子
        f_soc = self.f_f_soc2(self.f_f_soc1(soc))
        u1 = out[:,-1,0:1] - f_soc - out[:,-1,1:2] * self.R0
        weight = torch.ones_like(u1)
        d_u1 = torch.autograd.grad(u1,out,grad_outputs=weight,retain_graph=True,allow_unused=True,create_graph=True)[0][:,-1,-1]
        d_u1 = d_u1.reshape(batch_size,-1)
        loss2 = d_u1 + self.delta / (self.R1*self.C1) * u1 - out[:,-1,1:2] / self.C1  #第二个方程
        return soc,loss1,loss2



class MatDNNDataset(Dataset):
    def __init__(self, root , sequence_length):
        self.data = loadmat(root)
        self.sequence_length = sequence_length

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
        start = idx
        end = idx + self.sequence_length  # Exclusive
        return self.X[start:end, :], self.y[end - 1]

    def __len__(self):
        return len(self.y) - self.sequence_length + 1

def evaluate_pinn(test_x,test_y,net,loss_function):
    predict_y,_,_ = net(test_x)
    test_loss = loss_function(test_y,predict_y)
    return test_loss.item()


def evaluate_mlp(test_x,test_y,net,loss_function):
    predict_y = net(test_x)
    test_loss = loss_function(test_y,predict_y)
    return test_loss.item()


sequence_length = 20
batch_size = 256
num_epochs = 20
num_layers = 1
input_size = 4
hidden_size = 32
learning_rate = 0.001



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Fsmm(input_size,hidden_size,num_layers)
model_mlp = LSTM(input_size,hidden_size,num_layers,1,0.0)

model = torch.load('../pinn_lstm_saves LA92/model_pinn.pt')
model_mlp = torch.load('../pinn_lstm_saves LA92/model_lstm.pt')
validation_dataset = MatDNNDataset("../data/Panasonic 18650PF Data/0degC/Drive cycles/06-01-17_10.36 0degC_NN_Pan18650PF.mat",sequence_length)
validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=True)
loss = nn.MSELoss()
pinn_loss = 0
mlp_loss = 0
t = 0
for i,(x,y) in enumerate(validation_loader):
    x = x.to(device)
    y = y.to(device)
    pinn_pre_y,_,_ = model(x)
    mlp_y = model_mlp(x)
    pinn_loss += loss(pinn_pre_y,y).item()
    mlp_loss += loss(mlp_y,y).item()
    t += 1

print("pinn_loss:" ,pinn_loss/t,"mlp_loss:",mlp_loss/t)


model = torch.load('../pinn_lstm_saves 10 LA92/model_pinn.pt')
model_mlp = torch.load('../pinn_lstm_saves 10 LA92/model_lstm.pt')
validation_dataset = MatDNNDataset("../data/Panasonic 18650PF Data/10degC/Drive Cycles/03-27-17_09.06 10degC_NN_Pan18650PF.mat",sequence_length)
validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=True)
loss = nn.MSELoss()
pinn_loss = 0
mlp_loss = 0
t = 0
for i,(x,y) in enumerate(validation_loader):
    x = x.to(device)
    y = y.to(device)
    pinn_pre_y,_,_ = model(x)
    mlp_y = model_mlp(x)
    pinn_loss += loss(pinn_pre_y,y).item()
    mlp_loss += loss(mlp_y,y).item()
    t += 1

print("pinn_loss:" ,pinn_loss/t,"mlp_loss:",mlp_loss/t)

model = torch.load('../pinn_lstm_saves -10 LA92/model_pinn.pt').to(device)
model_mlp = torch.load('../pinn_lstm_saves -10 LA92/model_lstm.pt').to(device)

validation_dataset = MatDNNDataset("../data/Panasonic 18650PF Data/-10degC/Drive Cycles/06-14-17_13.12 n10degC_NN_Pan18650PF.mat",sequence_length)
validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=True)
loss = nn.MSELoss()
pinn_loss = 0
mlp_loss = 0
t = 0
for i,(x,y) in enumerate(validation_loader):
    x = x.to(device)
    y = y.to(device)
    pinn_pre_y,_,_ = model(x)
    mlp_y = model_mlp(x)
    pinn_loss += loss(pinn_pre_y,y).item()
    mlp_loss += loss(mlp_y,y).item()
    t += 1

print("pinn_loss:" ,pinn_loss/t,"mlp_loss:",mlp_loss/t)


model = torch.load('../pinn_lstm_saves -20 LA92/model_pinn.pt').to(device)
model_mlp = torch.load('../pinn_lstm_saves -20 LA92/model_lstm.pt').to(device)
validation_dataset = MatDNNDataset("../data/Panasonic 18650PF Data/-20degC/Drive cycles/06-23-17_23.35 n20degC_NN_Pan18650PF.mat",sequence_length)
validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=True)
loss = nn.MSELoss()
pinn_loss = 0
mlp_loss = 0
t = 0
for i,(x,y) in enumerate(validation_loader):
    x = x.to(device)
    y = y.to(device)
    pinn_pre_y,_,_ = model(x)
    mlp_y = model_mlp(x)
    pinn_loss += loss(pinn_pre_y,y).item()
    mlp_loss += loss(mlp_y,y).item()
    t += 1

print("pinn_loss:" ,pinn_loss/t,"mlp_loss:",mlp_loss/t)

