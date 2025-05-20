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
        num_layers,
        gaussian_noise=0.0,
        alpha=0.5,
        window=10,
    ):
        super(Fsmm, self).__init__()
        self.lstm = LSTM(input_size, hidden_size, num_layers, 1, gaussian_noise)
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
        out = x  # 输入形状: (batch_size, seq_len, features)
        batch_size, seq_len, dim = x.shape
        out.requires_grad_()

        soc = self.lstm(out)  # lstm

        lamada_i = self.delta / self.Q * out[:, -1, 1:2]  # 电流的函数
        weight = torch.ones_like(soc)
        d_soc = torch.autograd.grad(
            soc,
            out,
            grad_outputs=weight,
            retain_graph=True,
            allow_unused=True,
            create_graph=True,
        )[0][:, -1, -1]

        loss1 = lamada_i + d_soc  # 第一个式子

        f_soc = self.f_f_soc2(self.f_f_soc1(soc))

        Up_sequence = self._calculate_Up_sequence(out)
        loss2 = f_soc - out[:, -1, 0:1] - self.R0 * out[:, -1, 1:2] - Up_sequence[:, -1]

        return soc, loss1, loss2

    def _calculate_Up_sequence(self, out):

        batch_size, seq_len, _ = out.shape
        device = out.device
        Up_seq = torch.zeros(batch_size, seq_len, device=device)

        # 获取电压和时间序列
        voltage = out[:, :, 0]
        time = out[:, :, 3]

        for t in range(seq_len):
            start_idx = max(0, t - self.window + 1)
            voltage_window = voltage[:, start_idx : t + 1]  # (batch, window_size)

            pad_size = self.window - (t + 1 - start_idx)
            padded_voltage = torch.cat(
                [torch.zeros((batch_size, pad_size), device=device), voltage_window],
                dim=1,
            )

            time_window = time[:, start_idx : t + 1]
            valid_window = t + 1 - start_idx
            if valid_window > 1:
                time_diff = (time_window[:, -1] - time_window[:, 0]) / (
                    valid_window - 1
                )
            else:
                time_diff = torch.ones(batch_size, device=device)

            T_s_alpha = (time_diff**self.alpha).unsqueeze(-1)

            a = -T_s_alpha / (self.R1 * self.C1)
            I_k = out[:, t, 1]
            b = (T_s_alpha / self.C1) * I_k.unsqueeze(-1)

            history_sum = torch.einsum(
                "bw,w->b", padded_voltage, self.binom_coeffs[1 : self.window + 1]
            ).unsqueeze(-1)

            current_voltage = voltage[:, t].unsqueeze(-1)
            Up_seq[:, t] = (a * current_voltage + b - history_sum).squeeze()

        return Up_seq


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

sequence_length = 20
batch_size = 256
num_epochs = 100
num_layers = 1
input_size = 4
hidden_size = 32
learning_rate = 0.001
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Fsmm(input_size,hidden_size,num_layers)
model_mlp = LSTM(input_size,hidden_size,num_layers,1,gaussian_noise=0.0)
if torch.cuda.is_available():
    _ = torch.zeros(1, device='cuda')  # 显式初始化CUDA上下文
""" -20 deg"""

save_dir = 'pinn_fdiff/picture'
save_dir = os.path.join(save_dir)
model = torch.load("pinn_fdiff/pinn_lstm_saves -10 US06/model_pinn.pt", weights_only=False).to(
    device
)
model_mlp = torch.load(
    "pinn_fdiff/pinn_lstm_saves -10 US06/model_lstm.pt", weights_only=False
).to(device)
validation_dataset = MatDNNDataset(
    "data/Panasonic 18650PF Data/-10degC/Drive Cycles/06-07-17_08.39 n10degC_US06_Pan18650PF.mat",
    sequence_length,
)

a = list()
soc = list()
loss_fsmm = list()
loss_mlp = list()
y_fsmm = list()
y_mlp = list()
for i,u in enumerate(validation_dataset):
    x,y = u[0].to(device),u[1].to(device)
    soc.append(y.item())
    x = x.reshape(-1,sequence_length,4)
    a.append(x[0,sequence_length-1,-1].item())
    y_pre,_,_ = model(x)
    y_fsmm.append(y_pre.item())
    loss_fsmm.append(y_pre.item() - y.item())
    y_pre = model_mlp(x)
    loss_mlp.append(y_pre.item() - y.item())
    y_mlp.append(y_pre.item())
y_fsmm,y_mlp = np.array(y_fsmm),np.array(y_mlp)


fig,axes = plt.subplots(1,1,figsize=(28,20))
axes.plot(a,soc,color='b',label='Acture SOC')
axes.plot(a,y_fsmm,color='g',label='LSTM')
axes.plot(a,y_mlp,color='r',label='LSTM+PINN')
axes.legend()
plt.ylabel('SOC')
plt.xlabel('Time (s) T=0°C')
plt.savefig(
    os.path.join(save_dir, "soc_pred_-10deg US06_lstm.jpg"),
    bbox_inches="tight",
    pad_inches=0,
)


loss_fsmm,loss_mlp = np.array(loss_fsmm),np.array(loss_mlp)
fig,axes = plt.subplots(1,1,figsize=(28,20))
axes.plot(a,loss_fsmm,color='g',label='LSTM')
axes.plot(a,loss_mlp,color='r',label='LSTM+PINN')
axes.legend()
plt.ylabel('SOC error')
plt.xlabel('Time (s) T=0°C')

plt.savefig(
    os.path.join(save_dir, "soc_error_-10deg US06_lstm.jpg"),
    bbox_inches="tight",
    pad_inches=0,
)

np.save("pinn_fdiff/picture/US06_SOC_-10.npy", soc)
np.save("pinn_fdiff/picture/PINN_LSTM_US06_SOC_-10.npy", y_fsmm)
np.save("pinn_fdiff/picture/LSTM_US06_SOC_-10.npy", y_mlp)
np.save("pinn_fdiff/picture/LOSS_PINN_LSTM_US06_SOC_-10.npy", loss_fsmm)
np.save("pinn_fdiff/picture/LOSS_LSTM_US06_SOC_-10.npy", loss_mlp)
