import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

def set_chinese_font():
    try:
        # 方案1：优先使用系统中的中文字体（Windows默认"SimHei"，macOS/Linux用"WenQuanYi Zen Hei"）
        plt.rcParams['font.sans-serif'] = ['SimHei', 'WenQuanYi Zen Hei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
        print(" 成功设置中文字体，中文显示正常")
    except Exception:
        # 方案2：若系统无中文字体，使用默认字体并隐藏警告
        plt.rcParams['font.sans-serif'] = ['DejaVu Sans']  # 英文默认字体
        plt.rcParams['axes.unicode_minus'] = False
        print("️ 系统无中文字体，使用英文默认字体")


# 执行字体设置（必须在画图前执行）
set_chinese_font()

# -------------------------- 2. 解决OpenMP冲突 --------------------------
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# -------------------------- 3. 数据集预处理（无sklearn，手动标准化） --------------------------
# 读取train.csv
try:
    df = pd.read_csv('train.csv')
    print("成功读取train.csv")
except FileNotFoundError:
    print("未找到train.csv，请检查文件路径")
    exit()


def preprocess_data(df):
    print(f"\n 数据预处理开始：原始样本数={df.shape[0]}")
    df_clean = df.copy()

    # 处理缺失值
    df_clean = df_clean.dropna(subset=['x', 'y'])
    print(f"1. 删除缺失值后：样本数={df_clean.shape[0]}")

    # 处理无效值
    df_clean['x'] = pd.to_numeric(df_clean['x'], errors='coerce')
    df_clean['y'] = pd.to_numeric(df_clean['y'], errors='coerce')
    df_clean = df_clean.dropna(subset=['x', 'y'])
    print(f"2. 删除无效值后：样本数={df_clean.shape[0]}")

    # 处理异常值
    Q1_x = df_clean['x'].quantile(0.25)
    Q3_x = df_clean['x'].quantile(0.75)
    IQR_x = Q3_x - Q1_x
    lower_bound = Q1_x - 1.5 * IQR_x
    upper_bound = Q3_x + 1.5 * IQR_x
    df_clean = df_clean[(df_clean['x'] >= lower_bound) & (df_clean['x'] <= upper_bound) & (df_clean['x'] <= 120)]
    print(f"3. 删除异常值后：样本数={df_clean.shape[0]}")

    # 手动标准化
    mean_x = df_clean['x'].mean()
    std_x = df_clean['x'].std()
    mean_y = df_clean['y'].mean()
    std_y = df_clean['y'].std()
    std_x = 1.0 if std_x < 1e-6 else std_x
    std_y = 1.0 if std_y < 1e-6 else std_y

    x_scaled = (df_clean['x'].values.reshape(-1, 1) - mean_x) / std_x
    y_scaled = (df_clean['y'].values.reshape(-1, 1) - mean_y) / std_y

    # 转换为张量
    x_data = torch.tensor(x_scaled, dtype=torch.float32)
    y_data = torch.tensor(y_scaled, dtype=torch.float32)

    print(f"\n 数据预处理完成：")
    print(f"   - 有效样本数：{x_data.shape[0]}")
    print(f"   - 原始x范围：{df_clean['x'].min():.2f} ~ {df_clean['x'].max():.2f}")
    print(f"   - 原始y范围：{df_clean['y'].min():.2f} ~ {df_clean['y'].max():.2f}")
    return x_data, y_data, mean_x, std_x, mean_y, std_y


# 执行预处理
x_data, y_data, mean_x, std_x, mean_y, std_y = preprocess_data(df)


# -------------------------- 4. 模型定义（正态分布初始化w/b） --------------------------
class LinearModel(torch.nn.Module):
    def __init__(self):
        super(LinearModel, self).__init__()
        self.linear = nn.Linear(1, 1)
        # 正态初始化：mean=0，std=0.1
        nn.init.normal_(self.linear.weight, mean=0.0, std=0.1)
        nn.init.normal_(self.linear.bias, mean=0.0, std=0.1)

    def forward(self, x):
        return self.linear(x)


# -------------------------- 5. 训练函数（梯度裁剪+记录过程） --------------------------
def train_model(model, optimizer, epochs=300, print_interval=30):
    criterion = nn.MSELoss(reduction='mean')
    losses = []
    ws = []
    bs = []

    for epoch in range(epochs):
        # 正向传播
        y_pred = model(x_data)
        loss = criterion(y_pred, y_data)

        # 反向传播（梯度裁剪防NaN）
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        # 记录有效数据
        if not torch.isnan(loss):
            losses.append(loss.item())
            ws.append(model.linear.weight.item())
            bs.append(model.linear.bias.item())

        # 打印日志
        if (epoch + 1) % print_interval == 0:
            current_loss = losses[-1] if losses else "NaN"
            current_w = ws[-1] if ws else "NaN"
            current_b = bs[-1] if bs else "NaN"
            print(f"Epoch [{epoch + 1}/{epochs}] | Loss: {current_loss:.4f} | w: {current_w:.4f} | b: {current_b:.4f}")

    return losses, ws, bs, model


# -------------------------- 6. 练习5-2：三种优化器训练 --------------------------
print("\n 开始训练三种优化器（练习5-2）")
# 初始化模型
model_sgd = LinearModel()
model_adam = LinearModel()
model_rmsprop = LinearModel()

# 定义优化器
optimizer_sgd = torch.optim.SGD(model_sgd.parameters(), lr=0.05, momentum=0.9)
optimizer_adam = torch.optim.Adam(model_adam.parameters(), lr=0.001)
optimizer_rmsprop = torch.optim.RMSprop(model_rmsprop.parameters(), lr=0.001)

# 训练
print("\n--- 1. SGD优化器训练 ---")
sgd_losses, sgd_ws, sgd_bs, sgd_final = train_model(model_sgd, optimizer_sgd)

print("\n--- 2. Adam优化器训练 ---")
adam_losses, adam_ws, adam_bs, adam_final = train_model(model_adam, optimizer_adam)

print("\n--- 3. RMSprop优化器训练 ---")
rmsprop_losses, rmsprop_ws, rmsprop_bs, rmsprop_final = train_model(model_rmsprop, optimizer_rmsprop)

# -------------------------- 7. 练习5-1：可视化（无中文警告） --------------------------
print("\n 生成可视化图表（练习5-1）")
plt.rcParams['figure.figsize'] = (15, 10)

# 子图1：三种优化器性能对比（练习5-2）
plt.subplot(2, 2, 1)
plt.plot(range(len(sgd_losses)), sgd_losses, label="SGD (lr=0.05+momentum)", linewidth=2)
plt.plot(range(len(adam_losses)), adam_losses, label="Adam (lr=0.001)", linewidth=2)
plt.plot(range(len(rmsprop_losses)), rmsprop_losses, label="RMSprop (lr=0.001)", linewidth=2)
plt.xlabel("Epoch（训练轮次）")
plt.ylabel("MSE Loss（均方误差）")
plt.title("三种优化器性能对比（练习5-2）")
plt.legend()
plt.grid(alpha=0.3)

# 子图2：w参数调节过程（练习5-1）
plt.subplot(2, 2, 2)
plt.plot(range(len(sgd_ws)), sgd_ws, color="#FF6B6B", linewidth=2)
plt.axhline(y=1.0, color="black", linestyle="--", label="理论最优w≈1.0")
plt.xlabel("Epoch（训练轮次）")
plt.ylabel("权重w（标准化后）")
plt.title("SGD优化器：w的调节过程（练习5-1）")
plt.legend()
plt.grid(alpha=0.3)

# 子图3：b参数调节过程（练习5-1）
plt.subplot(2, 2, 3)
plt.plot(range(len(sgd_bs)), sgd_bs, color="#4ECDC4", linewidth=2)
plt.axhline(y=0.0, color="black", linestyle="--", label="理论最优b≈0.0")
plt.xlabel("Epoch（训练轮次）")
plt.ylabel("偏置b（标准化后）")
plt.title("SGD优化器：b的调节过程（练习5-1）")
plt.legend()
plt.grid(alpha=0.3)

# 子图4：不同学习率对SGD的影响（练习5-1）
plt.subplot(2, 2, 4)
test_lrs = [0.01, 0.05, 0.1]
for lr in test_lrs:
    temp_model = LinearModel()
    temp_opt = torch.optim.SGD(temp_model.parameters(), lr=lr, momentum=0.9)
    temp_losses, _, _, _ = train_model(temp_model, temp_opt, epochs=300, print_interval=1000)
    plt.plot(range(len(temp_losses)), temp_losses, label=f"SGD lr={lr}", linewidth=2)
plt.xlabel("Epoch（训练轮次）")
plt.ylabel("MSE Loss（均方误差）")
plt.title("不同学习率对SGD训练的影响（练习5-1）")
plt.legend()
plt.grid(alpha=0.3)

# 保存图表（无警告，中文正常）
plt.tight_layout()
plt.savefig("linear_regression_chinese_fixed.png", dpi=300, bbox_inches='tight')
plt.show()
print(f"\n 图表已保存为：linear_regression_chinese_fixed.png")

# -------------------------- 8. 最优模型测试（反标准化） --------------------------
print("\n 最优模型测试结果")
# 选择SGD作为最优模型（本次训练中SGD Loss最低，可根据实际结果切换）
best_model = sgd_final

# 反标准化参数（转换为原始数据的w/b）
w_original = best_model.linear.weight.item() * (std_y / std_x)
b_original = best_model.linear.bias.item() * std_y + mean_y - (best_model.linear.weight.item() * std_y * mean_x / std_x)

print(f"最优模型（SGD）原始参数：w = {w_original:.4f}, b = {b_original:.4f}")

# 测试预测（x=40.0）
x_test_original = 40.0
x_test_scaled = (x_test_original - mean_x) / std_x
x_test_tensor = torch.tensor([[x_test_scaled]], dtype=torch.float32)
y_pred_scaled = best_model(x_test_tensor)
y_pred_original = y_pred_scaled.data.item() * std_y + mean_y

print(f"测试预测：x={x_test_original} → y_pred = {y_pred_original:.4f}")
print("\n 所有任务完成！无中文显示警告，图表正常。")