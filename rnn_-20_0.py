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


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, gaussian_noise):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)
        self.noise = GaussianNoise(gaussian_noise)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)

        # Forward propagate RNN
        out, _ = self.rnn(x, h0)  # out: tensor of shape (batch_size, seq_length, hidden_size)

        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])
        return out

class Fsmm(nn.Module):
    def __init__(self,input_size,hidden_size,num_layers,gaussian_noise = 0.0):
        super(Fsmm, self).__init__()
        self.lstm = RNN(input_size,hidden_size,num_layers,1,gaussian_noise)
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

seed = 3#6727#random.randint(1,10000)
print(seed)
sequence_length = 20
batch_size = 256
num_epochs = 100
num_layers = 1
input_size = 4
hidden_size = 32
learning_rate = 0.001
save_dir = 'pinn_rnn_saves -20degC'
save_dir = os.path.join(save_dir)
setup_seed(seed)
if not os.path.exists(save_dir):
     os.mkdir(save_dir)


dataset_dir = "data/Panasonic 18650PF Data/-20degC/Drive cycles/"
train_files = [
               "06-24-17_04.29 n20degC_Cycle_1_Pan18650PF.mat",
               "06-24-17_11.58 n20degC_Cycle_2_Pan18650PF.mat",
               "06-24-17_19.29 n20degC_Cycle_3_Pan18650PF.mat",
               "06-25-17_03.01 n20degC_Cycle_4_Pan18650PF.mat"
              ]
train_datasets = list()
for f in train_files:
    temp = MatDNNDataset(dataset_dir + f,sequence_length)
    train_datasets.append(temp)

validation_dataset = MatDNNDataset("data/Panasonic 18650PF Data/-20degC/Drive cycles/06-23-17_23.35 n20degC_HWFET_Pan18650PF.mat",sequence_length)
train_loaders = list()
for d in train_datasets:
    temp = DataLoader(dataset=d, batch_size=batch_size, shuffle=True)
    train_loaders.append(temp)
validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=True)

model = Fsmm(input_size,hidden_size,num_layers).to(device)
model_rnn = RNN(input_size,hidden_size,num_layers,1,gaussian_noise=0.0).to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


pinn_epoch_loss = list()
epochs = list()
pinn_soc_loss = list()
mlp_epoch_loss = list()
#min_loss = 999
best_model = Fsmm(input_size,hidden_size,num_layers).to(device)
optimizer_lstm = torch.optim.Adam(model_rnn.parameters(),lr=learning_rate)
for epoch in range(num_epochs):
    pinn_single_loss = list()
    lstm_single_loss = list()
    total_train_loss = 0
    total_val_loss = 0
    num_train_batches = 0
    total_train_loss_mlp = 0
    need_loss = 0
    model.train()
    for seti, loader in enumerate(train_loaders):
        t_start = time.time()
        for i, (x, y) in enumerate(loader):
            t_load = time.time() - t_start
            t = time.time()
            with torch.backends.cudnn.flags(enabled=False):
               x = x.to(device)
               y = y.to(device)
               soc,loss1,loss2= model(x)
               loss = criterion(soc,y) + torch.mean(torch.square(loss1))
               loss_need = criterion(soc,y)
               soc_lstm = model_rnn(x)
               loss_lstm = criterion(soc_lstm, y)
               optimizer.zero_grad()
               loss.backward()
               optimizer.step()
               pinn_single_loss.append(loss_need.item())
               need_loss += loss_need.item()
               total_train_loss += loss.item()
               t_fw = time.time() - t

               optimizer_lstm.zero_grad()
               loss_lstm.backward()
               optimizer_lstm.step()

               # Backward and optimize
               # print(y)
               t = time.time()
               t_bk = time.time() - t
               t_start = time.time()

               lstm_single_loss.append(loss_lstm.item())
               total_train_loss_mlp += loss_lstm.item()
               num_train_batches += 1

               if i % 20 == 0:
                   print('Epoch [{}/{}], Seti [{}/{}]，Step [{}/{}],T_load: {:.4f}, T_fw: {:.4f}, T_bk: {:.4f}'
                           .format(epoch, num_epochs, seti, len(train_loaders), i, len(loader),
                                   t_load, t_fw, t_bk))
                   print("PINN_LSTM_mse:{:.5f}".format(loss_need.item()))
                   print("LSTM_mse:{:.5f}".format(loss_lstm.item()))
                   print("_______________________________")
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        ax.plot(range(len(lstm_single_loss)), pinn_single_loss, color='b', linewidth=1,
                label='pinn_soc_mse')
        ax.plot(range(len(lstm_single_loss)), lstm_single_loss, color='g', linewidth=1,
                label='rnn_soc_mse')
        ax.legend()
        plt.ylabel('loss')
        plt.savefig(os.path.join(save_dir, 'train{}.pdf'.format(epoch)), bbox_inches='tight', pad_inches=0)
        plt.close()
    averge_loss = total_train_loss / num_train_batches
    need_loss = need_loss / num_train_batches
    pinn_epoch_loss.append(averge_loss)
    pinn_soc_loss.append(need_loss)
    epochs.append(epoch+1)
    """
    if min_loss > need_loss:
        best_model = copy.deepcopy(model)
        min_loss = need_loss
    """
    averge_loss_mlp = total_train_loss_mlp / num_train_batches
    mlp_epoch_loss.append(averge_loss_mlp)

#model = copy.deepcopy(best_model)
loss = nn.MSELoss()
pinn_loss = list()
mlp_loss = list()
t = 0
model.train()
for i,(x,y) in enumerate(validation_loader):
    t += 1
    x = x.to(device)
    y = y.to(device)
    pinn_pre_y,_,_ = model(x)
    mlp_y = model_rnn(x)
    pinn_loss.append(loss(pinn_pre_y,y).item())
    mlp_loss.append(loss(mlp_y,y).item())
    x = x.to('cpu')
    y = y.to('cpu')


test = range(t)
total_pinn_loss = sum(pinn_loss)/t
total_loss = sum(mlp_loss)/t
print('pinn_loss:',total_pinn_loss,"rnn_loss:",total_loss)
torch.save(model, os.path.join(save_dir, 'model_pinn.pt'))
torch.save(model_rnn,os.path.join(save_dir,'model_rnn.pt'))
with open(os.path.join(save_dir, 'result.pkl'), 'wb') as f:
    pkl.dump({'epochs': epochs,
              'pinn_train_loss': pinn_soc_loss,
              'rnn_train_loss': mlp_epoch_loss,
              'pinn_loss':pinn_loss,
              'rnn_loss':mlp_loss}, f)


c = list([0,5,10])
for i,b in enumerate(c):
    fig,axes = plt.subplots(1,1,figsize=(20,14))

    axes.plot(epochs[b:],pinn_soc_loss[b:],color='b',linewidth=2, label='pinn_soc_loss')
    axes.plot(epochs[b:],mlp_epoch_loss[b:],color='g',linewidth=2,label='rnn')
    axes.legend()
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.title('the loss of y_predict and y_train')

    plt.savefig(os.path.join(save_dir, 'epochs{}-40.pdf'.format(b+1)), bbox_inches='tight', pad_inches=0)



fig,axes = plt.subplots(1,1,figsize=(20,14))
axes.plot(test,pinn_loss,color='b',label='pinn_soc_loss')
axes.plot(test,mlp_loss,color='g',label='rnn')
axes.legend()
plt.ylabel('loss')
plt.title('the loss of y_predict and y_test')
plt.savefig(os.path.join(save_dir, 'test40.pdf'), bbox_inches='tight', pad_inches=0)


