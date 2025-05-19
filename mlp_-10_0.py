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

def evaluate_pinn(test_x,test_y,net,loss_function):
    predict_y,_,_ = net(test_x)
    test_loss = loss_function(test_y,predict_y)
    return test_loss.item()


def evaluate_mlp(test_x,test_y,net,loss_function):
    predict_y = net(test_x)
    test_loss = loss_function(test_y,predict_y)
    return test_loss.item()

seed = 6727
print(seed)
batch_size = 256
num_epochs = 100
input_size = 4
hidden_size = 32
learning_rate = 0.001
save_dir = 'pinn_change_new_saves -10deg'
save_dir = os.path.join(save_dir)
setup_seed(seed)
if not os.path.exists(save_dir):
     os.mkdir(save_dir)


dataset_dir = "data/Panasonic 18650PF Data/-10degC/Drive cycles/"
train_files = [
               "06-10-17_11.25 n10degC_Cycle_1_Pan18650PF.mat",
               "06-10-17_18.35 n10degC_Cycle_2_Pan18650PF.mat",
               "06-11-17_01.39 n10degC_Cycle_3_Pan18650PF.mat",
               "06-11-17_08.42 n10degC_Cycle_4_Pan18650PF.mat"
              ]
train_datasets = list()
for f in train_files:
    temp = MatDNNDataset(dataset_dir + f)
    train_datasets.append(temp)

validation_dataset = MatDNNDataset("data/Panasonic 18650PF Data/-10degC/Drive Cycles/06-07-17_08.39 n10degC_HWFET_Pan18650PF.mat")
train_loaders = list()
for d in train_datasets:
    temp = DataLoader(dataset=d, batch_size=batch_size, shuffle=True)
    train_loaders.append(temp)
validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=True)

model = Fsmm(input_size,hidden_size).to(device)
model_mlp = NeuralNet(input_size,hidden_size,1).to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

pinn_epoch_loss = list()
epochs = list()
pinn_soc_loss = list()
mlp_epoch_loss = list()
test_x,test_y = validation_dataset[:]
test_x = test_x.to(device)
test_y = test_y.to(device)
zeros = torch.zeros_like(test_x[:,-1])
best_model = Fsmm(input_size,hidden_size).to(device)
optimizer_mlp = torch.optim.Adam(model_mlp.parameters(),lr=learning_rate)
for epoch in range(num_epochs):
    total_train_loss = 0
    total_val_loss = 0
    num_train_batches = 0
    need_loss = 0
    model.train()
    for loader in train_loaders:
        for i,(x,y) in enumerate(loader):
            x = x.to(device)
            y = y.to(device)
            soc,loss1,loss2= model(x)
            loss = criterion(soc,y) + torch.mean(torch.square(loss1)) + torch.mean(torch.square(loss2))
            loss_need = criterion(soc,y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            need_loss += loss_need.item()
            total_train_loss += loss.item()
            num_train_batches += 1
    averge_loss = total_train_loss / num_train_batches
    need_loss = need_loss / num_train_batches
    pinn_epoch_loss.append(averge_loss)
    pinn_soc_loss.append(need_loss)
    epochs.append(epoch+1)
    print("epoch:",epoch+1,
          "loss:",averge_loss,
          'soc_loss:',need_loss,
          )
    total_train_loss_mlp = 0
    num_train_batches = 0
    for loader in train_loaders:
        for i,(x,y) in enumerate(loader):
            x = x.to(device)
            y = y.to(device)
            soc_mlp = model_mlp(x)
            loss_mlp = criterion(soc_mlp,y)
            optimizer_mlp.zero_grad()
            loss_mlp.backward()
            optimizer_mlp.step()
            total_train_loss_mlp += loss_mlp.item()
            num_train_batches += 1
    averge_loss_mlp = total_train_loss_mlp / num_train_batches
    mlp_epoch_loss.append(averge_loss_mlp)
    print("epoch:",epoch+1,
          "loss:",averge_loss_mlp,
          )

loss = nn.MSELoss()
t_loss = nn.L1Loss()
pinn_loss = list()
mlp_loss = list()
p_loss = list()
m_loss = list()
t = 0
for i,(x,y) in enumerate(validation_loader):
    t += 1
    x = x.to(device)
    y = y.to(device)
    pinn_pre_y,_,_ = model(x)
    mlp_y = model_mlp(x)
    pinn_loss.append(loss(pinn_pre_y,y).item())
    mlp_loss.append(loss(mlp_y,y).item())
    p_loss.append(t_loss(pinn_pre_y,y).item())
    m_loss.append(t_loss(mlp_y,y).item())


test = range(t)
total_pinn_loss = sum(pinn_loss)/t
total_loss = sum(mlp_loss)/t
print('pinn_loss:',total_pinn_loss,"mlp_loss:",total_loss)
print('pinn_loss:',sum(p_loss)/t,"mlp_loss:",sum(m_loss)/t)
torch.save(model, os.path.join(save_dir, 'model_pinn.pt'))
torch.save(model_mlp,os.path.join(save_dir,'model_mlp.pt'))
with open(os.path.join(save_dir, 'result.pkl'), 'wb') as f:
    pkl.dump({'epochs': epochs,
              'pinn_train_loss': pinn_soc_loss,
              'mlp_train_loss': mlp_epoch_loss,
              'pinn_loss':pinn_loss,
              'mlp_loss':mlp_loss,
              'seed':seed,
              'p_loss':p_loss,
              'm_loss':m_loss}, f)

c = list([0,30,50,70])
for i,b in enumerate(c):
    fig,axes = plt.subplots(1,1,figsize=(28,20))

    axes.plot(epochs[b:],pinn_soc_loss[b:],color='b',linewidth=2, label='pinn_soc_loss')
    axes.plot(epochs[b:],mlp_epoch_loss[b:],color='g',linewidth=2,label='mlp')
    axes.legend()
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.title('the loss of y_predict and y_train')

    plt.savefig(os.path.join(save_dir, 'epochs{}-5.pdf'.format(b+1)), bbox_inches='tight', pad_inches=0)


fig,axes = plt.subplots(1,1,figsize=(28,20))
axes.plot(test,pinn_loss,color='b',label='pinn_soc_loss')
axes.plot(test,mlp_loss,color='g',label='mlp')
axes.legend()
plt.ylabel('loss')
plt.title('the loss of y_predict and y_test')

plt.savefig(os.path.join(save_dir, 'test.pdf'), bbox_inches='tight', pad_inches=0)

