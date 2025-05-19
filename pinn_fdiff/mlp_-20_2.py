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


def binomial_coeff(alpha: float, j: int) -> float:
    if j < 0:
        return 0.0
    coeff = 1.0
    for i in range(j):
        coeff *= (alpha - i) / (i + 1)
    return coeff


class Fsmm(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        gaussian_noise=0.0,
        alpha=0.5,
        window=10,
    ):
        super(Fsmm, self).__init__()
        # MLP模型替换LSTM (输入输出维度需调整)
        self.mlp = nn.Sequential(
            nn.Linear(input_size, hidden_size), nn.ReLU(), nn.Linear(hidden_size, 1)
        )
        self.Q = nn.Parameter(torch.tensor(2.9), requires_grad=True)
        self.delta = nn.Parameter(torch.tensor(1.0), requires_grad=True)
        self.noise = GaussianNoise(gaussian_noise)
        self.f_f_soc1 = nn.Linear(1, hidden_size)
        self.f_f_soc2 = nn.Linear(hidden_size, 1)
        self.U0 = nn.Parameter(torch.tensor(1.0), requires_grad=True)
        self.R0 = nn.Parameter(torch.tensor(1.0), requires_grad=True)
        self.C1 = nn.Parameter(torch.tensor(1.0), requires_grad=True)
        self.R1 = nn.Parameter(torch.tensor(1.0), requires_grad=True)
        self.alpha = alpha
        self.window = window

        # 历史记录缓冲区（需适配MLP的非序列输入）
        self.register_buffer("Up_history", torch.zeros((1, window)))
        self.register_buffer("time_history", torch.zeros((1, window)))
        self.register_buffer("binom_coeffs", self._init_binom_coeffs())

    def _init_binom_coeffs(self):
        coeffs = []
        for j in range(self.window + 1):
            coeff = (-1) ** j * binomial_coeff(self.alpha, j)
            coeffs.append(coeff)
        return torch.tensor(coeffs, dtype=torch.float32)

    def forward(self, x):
        """处理MLP的非序列输入（形状[batch, features]）"""
        batch_size = x.size(0)
        out = x.clone().requires_grad_(True)  # 确保可计算梯度

        # MLP前向传播 (直接处理当前时刻数据)
        soc = self.mlp(out)  # 形状[batch, 1]

        # 物理约束1：SOC变化率
        lamada_i = self.delta / self.Q * out[:, 1:2]  # 电流项
        weight = torch.ones_like(soc)
        d_soc = torch.autograd.grad(
            outputs=soc,
            inputs=out,
            grad_outputs=weight,
            retain_graph=True,
            create_graph=True,
        )[0][
            :, -1
        ]  # 取最后一个特征的梯度
        loss1 = lamada_i + d_soc.unsqueeze(-1)

        # 物理约束2：电压平衡
        f_soc = self.f_f_soc2(torch.relu(self.f_f_soc1(soc)))

        # 适配MLP的Up计算（需外部维护历史序列）
        Up = self._calculate_Up(out.unsqueeze(1))  # 临时增加序列维度
        loss2 = f_soc - out[:, 0:1] - self.R0 * out[:, 1:2] - Up

        return soc, loss1, loss2

    def _calculate_Up(self, out):
        """简化版Up计算（假设外部已维护电压序列）"""
        # out形状: [batch, 1, features]（模拟单时间步）
        batch_size = out.size(0)
        device = out.device

        # 获取当前电压和电流
        current_voltage = out[:, -1, 0]
        I_k = out[:, -1, 1]

        # 模拟时间差（MLP需外部提供时间信息）
        time_diff = torch.ones(batch_size, device=device)  # 默认单位时间差
        T_s_alpha = (time_diff**self.alpha).unsqueeze(-1)

        # 计算Up（简化版，实际使用时需维护真实历史窗口）
        a = -T_s_alpha / (self.R1 * self.C1)
        b = (T_s_alpha / self.C1) * I_k.unsqueeze(-1)

        # 模拟历史项（实际应用需用真实历史数据替换）
        history_sum = torch.zeros(batch_size, 1, device=device)

        Up = a * current_voltage.unsqueeze(-1) + b - history_sum
        return Up


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

seed = 100#6727
print(seed)
batch_size = 256
num_epochs = 100
input_size = 4
hidden_size = 32
learning_rate = 0.001
save_dir = 'pinn_fdiff/pinn_change_new_saves -20deg NN'
save_dir = os.path.join(save_dir)
setup_seed(seed)
if not os.path.exists(save_dir):
     os.mkdir(save_dir)


dataset_dir = "data/Panasonic 18650PF Data/-20degC/Drive cycles/"
train_files = [
               "06-23-17_23.35 n20degC_LA92_Pan18650PF.mat",
               "06-23-17_23.35 n20degC_HWFET_Pan18650PF.mat",
    "06-23-17_23.35 n20degC_UDDS_Pan18650PF.mat",
    "06-25-17_10.31 n20degC_US06_Pan18650PF.mat"
              ]
train_datasets = list()
for f in train_files:
    temp = MatDNNDataset(dataset_dir + f)
    train_datasets.append(temp)

validation_dataset = MatDNNDataset("data/Panasonic 18650PF Data/-20degC/Drive cycles/06-23-17_23.35 n20degC_NN_Pan18650PF.mat")
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
