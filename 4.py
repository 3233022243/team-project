import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# 设置中文字体和解决库冲突
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# 读取数据
print("正在读取数据...")
data = pd.read_csv('train.csv')

# 检查数据基本信息
print("数据基本信息:")
print(data.info())
print(f"\n数据形状: {data.shape}")

# 检查缺失值
print(f"\n缺失值统计:")
print(data.isnull().sum())

# 检查异常值（无穷大或NaN）
print(f"\n异常值检查:")
print(f"x列 - 无穷大: {np.isinf(data['x']).sum()}, NaN: {data['x'].isna().sum()}")
print(f"y列 - 无穷大: {np.isinf(data['y']).sum()}, NaN: {data['y'].isna().sum()}")

# 处理缺失值 - 如果有缺失值就用均值填充
if data.isnull().sum().sum() > 0:
    print("\n发现缺失值，使用均值填充...")
    data = data.fillna(data.mean())

# 移除可能的无穷大值
data = data.replace([np.inf, -np.inf], np.nan)
data = data.fillna(data.mean())

print(f"\n处理后的数据统计:")
print(f"x: 最小值={data['x'].min():.2f}, 最大值={data['x'].max():.2f}, 均值={data['x'].mean():.2f}")
print(f"y: 最小值={data['y'].min():.2f}, 最大值={data['y'].max():.2f}, 均值={data['y'].mean():.2f}")

# 准备数据
x_data = data['x'].values.reshape(-1, 1)
y_data = data['y'].values.reshape(-1, 1)

# 数据标准化（提高训练稳定性）
x_mean, x_std = x_data.mean(), x_data.std()
y_mean, y_std = y_data.mean(), y_data.std()

x_normalized = (x_data - x_mean) / x_std
y_normalized = (y_data - y_mean) / y_std

print(f"\n标准化后范围:")
print(f"x: [{x_normalized.min():.2f}, {x_normalized.max():.2f}]")
print(f"y: [{y_normalized.min():.2f}, {y_normalized.max():.2f}]")

# 转换为PyTorch张量
x_tensor = torch.FloatTensor(x_normalized)
y_tensor = torch.FloatTensor(y_normalized)


# 定义线性回归模型
class LinearRegression(nn.Module):
    def __init__(self):
        super(LinearRegression, self).__init__()
        self.linear = nn.Linear(1, 1)  # y = wx + b

    def forward(self, x):
        return self.linear(x)


# 初始化模型
model = LinearRegression()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# 存储训练历史
w_history = []
b_history = []
loss_history = []

print("\n开始训练线性回归模型 y = wx + b ...")

# 训练模型
epochs = 1000
for epoch in range(epochs):
    # 前向传播
    y_pred = model(x_tensor)
    loss = criterion(y_pred, y_tensor)

    # 反向传播
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # 记录参数和损失
    w = model.linear.weight.data.item()
    b = model.linear.bias.data.item()

    w_history.append(w)
    b_history.append(b)
    loss_history.append(loss.item())

    # 每100轮打印一次进度
    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.6f}')

# 将标准化后的参数转换回原始尺度
final_w_normalized = w_history[-1]
final_b_normalized = b_history[-1]

# 转换公式: y = w_norm * (x - x_mean)/x_std + b_norm
# 转换为原始尺度: y = (w_norm * y_std / x_std) * x + (b_norm * y_std + y_mean - w_norm * y_std * x_mean / x_std)
final_w_original = final_w_normalized * (y_std / x_std)
final_b_original = final_b_normalized * y_std + y_mean - final_w_original * x_mean

print(f"\n训练完成!")
print(f"标准化尺度参数: w = {final_w_normalized:.6f}, b = {final_b_normalized:.6f}")
print(f"原始尺度参数: w = {final_w_original:.4f}, b = {final_b_original:.4f}")
print(f"最终损失: {loss_history[-1]:.6f}")

# 可视化训练过程
plt.figure(figsize=(15, 10))

# 1. 权重w的变化
plt.subplot(2, 2, 1)
plt.plot(w_history, 'b-', linewidth=2)
plt.title('权重 (w) 变化过程', fontsize=14, fontweight='bold')
plt.xlabel('训练轮次 (Epoch)')
plt.ylabel('权重值')
plt.grid(True, alpha=0.3)
# 添加初始值和最终值标注
plt.axhline(y=w_history[0], color='r', linestyle='--', alpha=0.7, label=f'初始值: {w_history[0]:.4f}')
plt.axhline(y=w_history[-1], color='g', linestyle='--', alpha=0.7, label=f'最终值: {w_history[-1]:.4f}')
plt.legend()

# 2. 偏置b的变化
plt.subplot(2, 2, 2)
plt.plot(b_history, 'g-', linewidth=2)
plt.title('偏置 (b) 变化过程', fontsize=14, fontweight='bold')
plt.xlabel('训练轮次 (Epoch)')
plt.ylabel('偏置值')
plt.grid(True, alpha=0.3)
plt.axhline(y=b_history[0], color='r', linestyle='--', alpha=0.7, label=f'初始值: {b_history[0]:.4f}')
plt.axhline(y=b_history[-1], color='g', linestyle='--', alpha=0.7, label=f'最终值: {b_history[-1]:.4f}')
plt.legend()

# 3. 损失函数变化
plt.subplot(2, 2, 3)
plt.plot(loss_history, 'r-', linewidth=2)
plt.title('损失函数变化过程', fontsize=14, fontweight='bold')
plt.xlabel('训练轮次 (Epoch)')
plt.ylabel('损失值 (MSE)')
plt.yscale('log')  # 使用对数坐标更好地观察变化
plt.grid(True, alpha=0.3)
plt.legend(['损失曲线'])

# 4. 前100个epoch的详细视图
plt.subplot(2, 2, 4)
plt.plot(loss_history[:100], 'r-', linewidth=2)
plt.title('前100轮损失变化详情', fontsize=14, fontweight='bold')
plt.xlabel('训练轮次 (Epoch)')
plt.ylabel('损失值 (MSE)')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('linear_regression_training.png', dpi=300, bbox_inches='tight')
plt.show()

# 绘制拟合结果
plt.figure(figsize=(12, 8))

# 原始数据散点图
plt.scatter(x_data, y_data, alpha=0.6, s=30, label='数据点', color='blue')

# 生成拟合直线
x_range = np.linspace(x_data.min(), x_data.max(), 100)
x_range_normalized = (x_range - x_mean) / x_std
x_range_tensor = torch.FloatTensor(x_range_normalized.reshape(-1, 1))

with torch.no_grad():
    y_pred_normalized = model(x_range_tensor).numpy()

y_pred_original = y_pred_normalized * y_std + y_mean

plt.plot(x_range, y_pred_original, 'r-', linewidth=3,
         label=f'拟合直线: y = {final_w_original:.4f}x + {final_b_original:.4f}')

plt.xlabel('x', fontsize=12)
plt.ylabel('y', fontsize=12)
plt.title('线性回归拟合结果: y = wx + b', fontsize=14, fontweight='bold')
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)

# 添加公式文本
equation_text = f'y = {final_w_original:.4f}x + {final_b_original:.4f}\nR² = {1 - loss_history[-1]:.4f}'
plt.text(0.05, 0.95, equation_text, transform=plt.gca().transAxes,
         fontsize=12, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

plt.savefig('linear_regression_fit.png', dpi=300, bbox_inches='tight')
plt.show()

# 绘制参数收敛的3D视图
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# 绘制训练路径
ax.plot(w_history, b_history, loss_history, 'b-', linewidth=2, alpha=0.7, label='训练路径')
ax.scatter(w_history[0], b_history[0], loss_history[0], color='green', s=100, label='起点')
ax.scatter(w_history[-1], b_history[-1], loss_history[-1], color='red', s=100, label='终点')

ax.set_xlabel('权重 w', fontsize=12)
ax.set_ylabel('偏置 b', fontsize=12)
ax.set_zlabel('损失', fontsize=12)
ax.set_title('参数空间中的训练路径', fontsize=14, fontweight='bold')
ax.legend()

plt.savefig('training_path_3d.png', dpi=300, bbox_inches='tight')
plt.show()

# 打印详细的训练统计
print(f"\n=== 训练统计 ===")
print(f"训练轮次: {epochs}")
print(f"初始损失: {loss_history[0]:.6f}")
print(f"最终损失: {loss_history[-1]:.6f}")
print(
    f"损失减少: {loss_history[0] - loss_history[-1]:.6f} ({((loss_history[0] - loss_history[-1]) / loss_history[0] * 100):.1f}%)")
print(f"权重变化: {w_history[0]:.6f} → {w_history[-1]:.6f}")
print(f"偏置变化: {b_history[0]:.6f} → {b_history[-1]:.6f}")
print(f"最终模型: y = {final_w_original:.4f}x + {final_b_original:.4f}")

# 验证模型
with torch.no_grad():
    test_predictions = model(x_tensor)
    test_loss = criterion(test_predictions, y_tensor)
    print(f"验证损失: {test_loss.item():.6f}")

print("\n所有图表已保存为PNG文件!")