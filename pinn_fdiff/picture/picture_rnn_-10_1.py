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
mpl.use('Agg')
import matplotlib.pyplot as plt

import  time
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
        self.lstm = RNN(input_size, hidden_size, num_layers, 1, gaussian_noise)
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
            
            past_Up = Up_seq[:,0:t]  # (batch, window_size)
            valid_window = past_Up.shape[1] + 1
            pad_size = max(self.window - valid_window + 1, 0)


            time_window = time[:, 0 : t + 1]
            if time_window.size(1) > 1:
                time_diff = (time_window[:, -1] - time_window[:, 0]) / (
                    time_window.size(1) - 1
                )
            else:
                time_diff = torch.ones(batch_size, device=device)
            if pad_size > 0:
                # 前面补零
                zeros = torch.zeros((batch_size, pad_size), device=device)
                history_buffer = torch.cat([zeros, past_Up], dim=1)
            else:
                # 已经达到 window 长度，直接使用过去的 Up_seq 切片
                history_buffer = past_Up[:, -self.window :]

            T_s_alpha = (time_diff**self.alpha).unsqueeze(-1)

            a = -T_s_alpha / (self.R1 * self.C1)
            I_k = out[:, t, 1]
            b = (T_s_alpha / self.C1) * I_k.unsqueeze(-1)

            history_sum = torch.einsum(
                "bw,w->b", history_buffer, self.binom_coeffs[1 : self.window + 1]
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

def evaluate_pinn(test_x,test_y,net,loss_function):
    predict_y,_,_ = net(test_x)
    test_loss = loss_function(test_y,predict_y)
    return test_loss.item()


def evaluate_mlp(test_x,test_y,net,loss_function):
    predict_y = net(test_x)
    test_loss = loss_function(test_y,predict_y)
    return test_loss.item()
# 设置设备和参数

torch.serialization.add_safe_globals([Fsmm, RNN, GaussianNoise])
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
sequence_length = 20  # 与训练时保持一致
save_dir = 'pinn_fdiff/pinn_rnn_saves -10degC LA92'  # 模型保存路径
test_data_path = "data/Panasonic 18650PF Data/-10degC/Drive Cycles/06-07-17_08.39 n10degC_HWFET_Pan18650PF.mat"  # 替换为测试数据路径

# 加载训练好的模型
model_pinn = torch.load(os.path.join(save_dir, 'model_pinn.pt'), map_location=device,weights_only=False)
model_rnn = torch.load(os.path.join(save_dir, 'model_rnn.pt'), map_location=device,weights_only=False)
model_pinn.train()
model_rnn.train()

# 加载测试数据集
test_dataset = MatDNNDataset("data/Panasonic 18650PF Data/-10degC/Drive Cycles/06-07-17_08.39 n10degC_HWFET_Pan18650PF.mat", sequence_length)
test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)

# 收集数据
actual_soc = []
pinn_pred = []
rnn_pred = []
timestamps = []

loss_a = nn.L1Loss()
loss = nn.MSELoss()
pinn_loss = list()
mlp_loss = list()
p_loss = list()
m_loss = list()
t = 0
for i,(x,y) in enumerate(test_loader):
    t += 1
    x = x.to(device)
    y = y.to(device)
    soc_pinn, _, _ = model_pinn(x)
    soc_rnn = model_rnn(x)

    pinn_loss.append(loss(soc_pinn,y).item())
    mlp_loss.append(loss(soc_rnn,y).item())
    p_loss.append(loss_a(soc_pinn,y).item())
    m_loss.append(loss_a(soc_rnn,y).item())
    actual_soc.extend(y.detach().cpu().numpy().flatten())
    pinn_pred.extend(soc_pinn.detach().cpu().numpy().flatten())
    rnn_pred.extend(soc_rnn.detach().cpu().numpy().flatten())
    timestamps.extend(x[:, -1, 3].detach().cpu().numpy().flatten())

test = range(t)
total_pinn_loss = sum(pinn_loss)/t
total_loss = sum(mlp_loss)/t
print('pinn_loss:',total_pinn_loss,"rnn_loss:",total_loss)
print('pinn_loss:',sum(p_loss)/t,"rnn_loss:",sum(m_loss)/t)

# 转换为numpy数组
actual_soc = np.array(actual_soc)
pinn_pred = np.array(pinn_pred)
rnn_pred = np.array(rnn_pred)
timestamps = np.array(timestamps)

# 创建绘图
plt.figure(figsize=(15, 8))
plt.plot(timestamps, actual_soc, label='Actual SOC', linewidth=2, alpha=0.8)
plt.plot(timestamps, pinn_pred, label='PINN Prediction', linestyle='--', linewidth=1.5)
plt.plot(timestamps, rnn_pred, label='RNN Prediction', linestyle='-.', linewidth=1.5)

# 添加图例和标签
plt.xlabel('Time (s)', fontsize=12)
plt.ylabel('State of Charge (SOC)', fontsize=12)
plt.title('SOC Prediction Comparison', fontsize=14)
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)
plt.tight_layout()

# 保存图像
plot_path = os.path.join(save_dir, 'soc_prediction_comparison.png')
plt.savefig(plot_path, dpi=300, bbox_inches='tight')
plt.close()

print(f"Plot saved to {plot_path}")