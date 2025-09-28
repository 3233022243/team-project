import torch
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib

matplotlib.rcParams["font.family"] = ["sans-serif"]
matplotlib.rcParams["font.sans-serif"] = ["SimHei"]  # 直接指定黑体

# 1. 读取并预处理数据
df = pd.read_csv('train.csv')
df = df.dropna(subset=['y'])  # 移除y列空值
x_data = df['x'].values
y_data = df['y'].values

# 数据标准化（加速收敛）
x_mean, x_std = x_data.mean(), x_data.std()
y_mean, y_std = y_data.mean(), y_data.std()
x_data_norm = (x_data - x_mean) / x_std if x_std != 0 else x_data
y_data_norm = (y_data - y_mean) / y_std if y_std != 0 else y_data

# 数据信息输出
print("数据信息：")
print(f"x范围：[{x_data.min():.2f}, {x_data.max():.2f}]，均值：{x_mean:.2f}")
print(f"y范围：[{y_data.min():.2f}, {y_data.max():.2f}]，均值：{y_mean:.2f}")
print(f"有效样本数：{len(x_data)}\n")

# 2. 初始化模型参数
w = torch.tensor([0.0], requires_grad=True)
b = torch.tensor([0.0], requires_grad=True)


# 3. 定义模型和损失函数
def forward(x):
    return w * x + b


def loss_func(y_pred, y_true):
    return (y_pred - y_true) ** 2


# 4. 训练配置
epoch_num = 2000
learning_rate = 0.005
w_history = []
b_history = []
loss_history = []

# 5. 训练过程
for epoch in range(epoch_num):
    total_loss = 0.0
    for x, y in zip(x_data_norm, y_data_norm):
        x_tensor = torch.tensor(x, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.float32)

        y_pred = forward(x_tensor)
        loss = loss_func(y_pred, y_tensor)
        loss.backward()

        # 梯度裁剪与参数更新
        with torch.no_grad():
            w_grad = torch.clamp(w.grad, -1.0, 1.0)
            b_grad = torch.clamp(b.grad, -1.0, 1.0)
            w -= learning_rate * w_grad
            b -= learning_rate * b_grad

        w.grad.zero_()
        b.grad.zero_()
        total_loss += loss.item()

    avg_loss = total_loss / len(x_data_norm)
    loss_history.append(avg_loss)
    w_history.append(w.item())
    b_history.append(b.item())

    if (epoch + 1) % 200 == 0:
        print(f"迭代 {epoch + 1}/{epoch_num} | w: {w.item():.4f} | b: {b.item():.4f} | 平均损失: {avg_loss:.6f}")

# 6. 还原参数到原始尺度
w_original = w.item() * (y_std / x_std) if x_std != 0 else w.item()
b_original = (b.item() * y_std) + y_mean - (w_original * x_mean)


# 7. 预测结果
def forward_original(x):
    return w_original * x + b_original


print("\n训练结果：")
print(f"原始数据模型：y = {w_original:.4f}*x + {b_original:.4f}")
print(f"x=4时的预测值：{forward_original(4):.4f}")

# 8. 可视化
plt.figure(figsize=(12, 10))

# 子图1：权重和偏置变化
plt.subplot(2, 1, 1)
plt.plot(range(1, epoch_num + 1), w_history, 'b-', label='标准化权重w')
plt.plot(range(1, epoch_num + 1), b_history, 'g-', label='标准化偏置b')
plt.xlabel('迭代次数')
plt.ylabel('参数值')
plt.title('权重和偏置的训练过程')
plt.legend()
plt.grid(linestyle='--', alpha=0.7)
plt.xlim(0, epoch_num)

# 子图2：拟合曲线与原始数据
plt.subplot(2, 1, 2)
plt.scatter(x_data, y_data, c='gray', alpha=0.6, label='原始数据')
x_min, x_max = x_data.min(), x_data.max()
x_range = torch.linspace(x_min, x_max, 200)
y_pred_range = forward_original(x_range.numpy())
plt.plot(x_range.numpy(), y_pred_range, 'r-', linewidth=2,
         label=f'拟合曲线: y={w_original:.2f}x + {b_original:.2f}')
plt.xlim(x_min - 0.1 * (x_max - x_min), x_max + 0.1 * (x_max - x_min))
plt.ylim(y_data.min() - 0.1 * (y_data.max() - y_data.min()), y_data.max() + 0.1 * (y_data.max() - y_data.min()))
plt.xlabel('x')
plt.ylabel('y')
plt.title('模型拟合效果')
plt.legend()
plt.grid(linestyle='--', alpha=0.7)

plt.tight_layout()
plt.show()