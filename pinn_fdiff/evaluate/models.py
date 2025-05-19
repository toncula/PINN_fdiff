import torch
import torch.nn as nn


class GaussianNoise(nn.Module):
    def __init__(self, stddev):
        super().__init__()
        self.stddev = stddev

    def forward(self, din):
        if self.training:
            return din + torch.autograd.Variable(torch.randn(din.size()) * self.stddev)
        return din

# Fully connected neural network with one hidden layer
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, gaussian_noise=0.0):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc3 = nn.Linear(hidden_size, num_classes)
        self.noise = GaussianNoise(gaussian_noise)

    def forward(self, x):
        out = x
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        return out

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, gaussian_noise):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)
        self.noise = GaussianNoise(gaussian_noise)

    def forward(self, x):
        # Set initial hidden and cell states
        x = self.noise(x)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)

        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_size)

        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])
        return out

class GRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, gaussian_noise):
        super(GRU, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)
        self.noise = GaussianNoise(gaussian_noise)

    def forward(self, x):
        # Set initial hidden and cell states
        x = self.noise(x)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)

        # Forward propagate GRU
        out, _ = self.gru(x, h0)  # out: tensor of shape (batch_size, seq_length, hidden_size)

        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])
        return out

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, gaussian_noise):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)
        self.noise = GaussianNoise(gaussian_noise)

    def forward(self, x):
        # Set initial hidden and cell states
        x = self.noise(x)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)

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


class Fsmm_rnn(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        num_layers,
        gaussian_noise=0.0,
        alpha=0.5,
        window=10,
    ):
        super(Fsmm_rnn, self).__init__()
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


class Fsmm_mlp(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        gaussian_noise=0.0,
        alpha=0.5,
        window=10,
    ):
        super(Fsmm_mlp, self).__init__()
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


class Fsmm_lstm(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        num_layers,
        gaussian_noise=0.0,
        alpha=0.5,
        window=10,
    ):
        super(Fsmm_lstm, self).__init__()
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
