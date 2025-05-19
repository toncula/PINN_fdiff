import pandas as pd
import torch.nn.functional as F
from mat4py import loadmat
import copy
import random

import os
import pickle as pkl

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
import matplotlib as mpl

mpl.use("TkAgg")
import matplotlib.pyplot as plt

import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# === 根据模型类型返回不同 Fsmm 实现 ===
def get_fsmm(model_type, *args, **kwargs):
    if model_type == "PINN_LSTM":
        from models import Fsmm_lstm as Fsmm

        return Fsmm(*args, **kwargs)
    elif model_type == "PINN_RNN":
        from models import Fsmm_rnn as Fsmm

        return Fsmm(*args, **kwargs)
    elif model_type == "PINN_MLP":
        from models import Fsmm_mlp as Fsmm 

        return Fsmm(*args, **kwargs)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")


class MatDNNDataset(Dataset):
    def __init__(self, root, sequence_length):
        self.data = loadmat(root)
        self.sequence_length = sequence_length

        self.BATTERY_AH_CAPACITY = 2.9000

        self.df = pd.DataFrame(self.data).T.apply(lambda x: pd.Series(x[0]))
        self.df = self.df.applymap(lambda x: x[0])

        for col in ["Chamber_Temp_degC", "TimeStamp", "Power", "Wh"]:
            self.df.pop(col)

        ah = self.df.pop("Ah")
        self.df["SOC"] = 1 + (ah / self.BATTERY_AH_CAPACITY)

        self.V = torch.tensor(self.df["Voltage"].values, dtype=torch.float32).reshape(
            -1, 1
        )
        self.I = torch.tensor(self.df["Current"].values, dtype=torch.float32).reshape(
            -1, 1
        )
        self.T = torch.tensor(
            self.df["Battery_Temp_degC"].values, dtype=torch.float32
        ).reshape(-1, 1)
        self.y = torch.tensor(self.df["SOC"].values, dtype=torch.float32).reshape(-1, 1)
        self.time = torch.tensor(self.df["Time"].values, dtype=torch.float32).reshape(
            -1, 1
        )
        self.X = torch.cat((self.V, self.I, self.T, self.time), dim=1)

    def __getitem__(self, idx):
        start = idx
        end = idx + self.sequence_length
        return self.X[start:end, :], self.y[end - 1]

    def __len__(self):
        return len(self.y) - self.sequence_length + 1


def compute_mae(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))


def compute_mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)


def evaluate(model, dataset, sequence_length, is_mlp=False):
    model.eval()
    preds, trues, times = [], [], []

    with torch.no_grad():
        for i in range(len(dataset)):
            x, y = dataset[i]
            x, y = x.to(device), y.item()
            trues.append(y)
            if x.ndim == 2:
                times.append(x[-1, -1].item())
            elif x.ndim == 1:
                times.append(x[-1].item())  # 只有一个时间步或已压缩成 1D
            else:
                raise ValueError(f"Unexpected x shape: {x.shape}")


            if sequence_length > 1:
                x = x.view(1, sequence_length, -1)
            else:
                x = x.view(1, -1)

            y_pred = model(x)[0] if not is_mlp else model(x)
            preds.append(y_pred.item())

    return (
        np.array(trues[-len(preds) :]),
        np.array(preds),
        np.array(times[-len(preds) :]),
    )


# ========== 配置 ==========
seq_len_mlp, seq_len_other = 1, 20
model_types = ["MLP", "PINN_MLP", "LSTM", "PINN_LSTM", "RNN", "PINN_RNN"]
model_paths = {
    "MLP": "pinn_fdiff/pinn_change_new_saves/model_mlp.pt",
    "PINN_MLP": "pinn_fdiff/pinn_change_new_saves/model_pinn.pt",
    "LSTM": "pinn_fdiff/pinn_lstm_saves/model_lstm.pt",
    "PINN_LSTM": "pinn_fdiff/pinn_lstm_saves/model_pinn.pt",
    "RNN": "pinn_fdiff/pinn_rnn_saves/model_rnn.pt",
    "PINN_RNN": "pinn_fdiff/pinn_rnn_saves/model_pinn.pt",
}

results = {}
target_file = "data/Panasonic 18650PF Data/0degC/Drive cycles/06-02-17_10.43 0degC_HWFET_Pan18650PF.mat"
plt.figure(figsize=(14, 8))

for model_type in model_types:
    path = model_paths[model_type]
    seq_len = seq_len_mlp if "MLP" in model_type else seq_len_other
    is_mlp = model_type in ["MLP"]

    dataset = MatDNNDataset(target_file, seq_len)
    model = torch.load(path, weights_only=False, map_location=device)

    y_true, y_pred, time_seq = evaluate(model, dataset, seq_len, is_mlp=is_mlp)

    mae = compute_mae(y_true, y_pred)
    mse = compute_mse(y_true, y_pred)
    results[model_type] = {"MAE": mae, "MSE": mse}

    plt.plot(time_seq, y_pred, label=model_type)

# 绘制真值
dataset_full = MatDNNDataset(target_file, 1)
y_true_full = [dataset_full[i][1].item() for i in range(len(dataset_full))]
t_full = [dataset_full[i][0][-1].item() for i in range(len(dataset_full))]
plt.plot(t_full, y_true_full, "k--", label="True SOC")

plt.xlabel("Time (s)")
plt.ylabel("SOC")
plt.title("SOC Prediction Across Models")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("all_model_predictions.jpg")

print("Evaluation Results:")
for name, metrics in results.items():
    print(f"{name:10s} | MAE = {metrics['MAE']:.4f} | MSE = {metrics['MSE']:.4f}")
