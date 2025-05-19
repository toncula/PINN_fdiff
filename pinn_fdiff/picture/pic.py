import matplotlib.pyplot as plt
import numpy as np

fdiff_lstm = np.load("pinn_fdiff/picture/PINN_LSTM_US06_SOC_-10.npy")
lstm = np.load("pinn_fdiff/picture/LSTM_US06_SOC_-10.npy")
fdiff_rnn = np.load("pinn_fdiff/picture/PINN_RNN_US06_SOC_-10.npy")
rnn = np.load("pinn_fdiff/picture/RNN_US06_SOC_-10.npy")
fdiff_mlp = np.load("pinn_fdiff/picture/PINN_MLP_US06_SOC_-10.npy")
mlp = np.load("pinn_fdiff/picture/MLP_US06_SOC_-10.npy")
soc = np.load("pinn_fdiff/picture/US06_SOC_-10.npy")
# 找到最小长度
min_len = min(
    len(soc),
    len(mlp),
    len(fdiff_mlp),
    len(rnn),
    len(fdiff_rnn),
    len(lstm),
    len(fdiff_lstm),
)

# 截断所有数组
a = np.arange(min_len)
soc = soc[:min_len]
mlp = mlp[:min_len]
fdiff_mlp = fdiff_mlp[:min_len]
rnn = rnn[:min_len]
fdiff_rnn = fdiff_rnn[:min_len]
lstm = lstm[:min_len]
fdiff_lstm = fdiff_lstm[:min_len]

fig, axes = plt.subplots(1, 1, figsize=(28, 20))

# 绘图
fig, axes = plt.subplots(1, 1, figsize=(28, 20))
axes.plot(a, soc, label="True SOC", color="black", linewidth=3)
axes.plot(a, mlp, label="MLP", linestyle="--")
axes.plot(a, fdiff_mlp, label="FDIFF-MLP")
axes.plot(a, rnn, label="RNN", linestyle="--")
axes.plot(a, fdiff_rnn, label="FDIFF-RNN")
axes.plot(a, lstm, label="LSTM", linestyle="--")
axes.plot(a, fdiff_lstm, label="FDIFF-LSTM")

axes.set_xlabel("Time (s)", fontsize=18)
axes.set_ylabel("SOC", fontsize=18)
axes.set_title("SOC Prediction Comparison on HWEFT at 0°C", fontsize=22)
axes.legend(fontsize=16)
axes.grid(True)

# 保存图像
plt.tight_layout()
plt.savefig("pinn_fdiff/picture/HWEFT_SOC_comparison_0degC.jpg", dpi=300)

# 显示图像（可选）
plt.show()
