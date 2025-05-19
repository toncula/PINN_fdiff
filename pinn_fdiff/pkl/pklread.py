import os
import pickle
import pandas as pd

records = []

# 遍历所有 .pkl 文件
for fname in os.listdir("./pinn_fdiff/pkl"):
    if not fname.endswith(".pkl"):
        continue
    method, dataset, temp_str = fname.replace(".pkl", "").split("_")
    temp = int(temp_str)

    with open(os.path.join("./pinn_fdiff/pkl", fname), "rb") as f:
        dat = pickle.load(f)
    print("Available keys in this pkl:", fname, list(dat.keys()))

    # 使用最统一的 key 命名进行处理
    def mean_or_zero(key_list):
        """支持多个可能 key 名的平均值提取"""
        for key in key_list:
            vals = dat.get(key, [])
            if isinstance(vals, list) and len(vals) > 0:
                try:
                    return float(sum(vals)) / len(vals)
                except Exception:
                    continue
        return 0.0

    pinn_mae = mean_or_zero(["pinn_mae_loss","p_loss"])
    pinn_mse = mean_or_zero(["pinn_loss"])
    mlp_mae = mean_or_zero(["rnn_mae_loss", "m_loss"])  # 统一为“MLP”类
    mlp_mse = mean_or_zero(["mlp_loss", "rnn_loss"])

    records.append(
        {
            "Dataset": dataset,
            "Method": method,
            "Temp": temp,
            "PINN MAE": pinn_mae,
            "PINN MSE": pinn_mse,
            "MLP MAE": mlp_mae,
            "MLP MSE": mlp_mse,
        }
    )

# 构造 DataFrame
df = pd.DataFrame(records)


# 保留三位有效数字
def format_sig(x, sig=3):
    if pd.isna(x):
        return "-"
    return f"{x:.{sig}g}"


for col in ["PINN MAE", "PINN MSE", "MLP MAE", "MLP MSE"]:
    df[col] = df[col].apply(format_sig)

df = df.sort_values(by=["Dataset", "Temp", "Method"])
cols = ["Dataset", "Temp", "Method", "PINN MAE", "PINN MSE", "MLP MAE", "MLP MSE"]
df = df[cols]

# 输出 LaTeX 表格
with open("all_results_three_sig.tex", "w", encoding="utf-8") as f:
    f.write(r"\begin{table*}[htbp]" + "\n")
    f.write(r"\centering" + "\n")
    f.write(
        r"\caption{Final MAE and MSE of all methods across datasets and temperatures (3 significant digits).}"
        + "\n"
    )
    f.write(r"\label{tab:all_results_3sig}" + "\n")
    f.write(r"\begin{tabular}{lllcccc}" + "\n")
    f.write(r"\toprule" + "\n")
    f.write(
        "Dataset & Temp ($^\circ$C) & Method & PINN MAE & PINN MSE & MLP MAE & MLP MSE \\\\"
        + "\n"
    )
    f.write(r"\midrule" + "\n")
    for _, row in df.iterrows():
        line = " & ".join(str(row[col]) for col in df.columns) + r" \\"
        f.write(line + "\n")
    f.write(r"\bottomrule" + "\n")
    f.write(r"\end{tabular}" + "\n")
    f.write(r"\end{table*}" + "\n")

print("✅ 合并格式并输出 LaTeX 表格成功：all_results_three_sig.tex")
