# 1. 优先解决OpenMP冲突（必须放在最顶部）
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# 2. 导入所需库
import torch
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np  # 新增：用于数据归一化计算

# 设置中文字体（避免中文乱码）
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示异常

# -------------------------- 配置参数（已调整学习率） --------------------------
FILE_PATH = r'C:\Users\dell\Desktop\train.csv'  # 你的数据集路径
NUM_EPOCHS = 150
LEARNING_RATE = 0.001  # 降低学习率，避免梯度爆炸
INITIAL_WEIGHT = 0.01  # 初始权重（按需求设置）
INITIAL_BIAS = 0.0  # 初始偏置


# -------------------------- 数据加载与预处理（新增数据归一化） --------------------------
def load_data(path):
    """加载数据+处理缺失值+数据归一化+转换为张量"""
    if not os.path.exists(path):
        raise FileNotFoundError(f"文件不存在: {path}")

    # 读取CSV数据
    df = pd.read_csv(path)
    # 只保留数值型列（排除非数值特征）
    numeric_cols = df.select_dtypes(include=['number']).columns
    if len(numeric_cols) < 2:
        raise ValueError("数据集至少需要2列数值型数据（1列特征+1列标签）")

    # 用均值填充缺失值
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())

    # 提取特征（第1列数值）和标签（第2列数值）
    features_np = df[numeric_cols].iloc[:, 0].values  # 特征：numpy数组
    labels_np = df[numeric_cols].iloc[:, 1].values  # 标签：numpy数组

    # -------------------------- 核心修复：数据归一化（避免梯度爆炸） --------------------------
    # Min-Max归一化：将数据缩放到 [0, 1] 范围
    feat_min, feat_max = features_np.min(), features_np.max()
    label_min, label_max = labels_np.min(), labels_np.max()

    # 避免分母为0（若所有值相同，直接设为0）
    if feat_max - feat_min != 0:
        features_np = (features_np - feat_min) / (feat_max - feat_min)
    else:
        features_np = np.zeros_like(features_np)

    if label_max - label_min != 0:
        labels_np = (labels_np - label_min) / (label_max - label_min)
    else:
        labels_np = np.zeros_like(labels_np)

    # 转换为PyTorch张量（reshape(-1,1)确保是二维张量：[样本数, 特征数]）
    features = torch.tensor(features_np, dtype=torch.float32).reshape(-1, 1)
    labels = torch.tensor(labels_np, dtype=torch.float32).reshape(-1, 1)

    print(f"数据处理完成：")
    print(f"  - 特征形状: {features.shape}, 标签形状: {labels.shape}")
    print(f"  - 特征范围（归一化后）: [{features.min():.4f}, {features.max():.4f}]")
    print(f"  - 标签范围（归一化后）: [{labels.min():.4f}, {labels.max():.4f}]")

    # 返回：张量+归一化参数（后续若需还原真实值可使用）
    return features, labels, (feat_min, feat_max, label_min, label_max)


# -------------------------- 加载数据（捕获异常，避免程序崩溃） --------------------------
try:
    X, y, norm_params = load_data(FILE_PATH)  # X=特征张量, y=标签张量
except Exception as e:
    print(f"数据加载失败: {e}")
    exit(1)


# -------------------------- 线性回归模型定义（保持原逻辑） --------------------------
class LinearModel(nn.Module):
    def __init__(self, init_w, init_b):
        super().__init__()
        self.linear = nn.Linear(1, 1)  # 1输入特征 → 1输出预测值

        # 用指定初始值初始化权重和偏置
        with torch.no_grad():  # 临时禁用梯度计算（避免初始化影响梯度）
            self.linear.weight.copy_(torch.tensor([[init_w]], dtype=torch.float32))
            self.linear.bias.copy_(torch.tensor([init_b], dtype=torch.float32))

    def forward(self, x):
        return self.linear(x)  # 前向传播：y = w*x + b


# -------------------------- 初始化模型、损失函数、优化器 --------------------------
model = LinearModel(INITIAL_WEIGHT, INITIAL_BIAS)  # 初始化模型
criterion = nn.MSELoss()  # 均方误差损失（适合回归任务）
optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)  # 随机梯度下降

# -------------------------- 训练过程（跟踪参数和损失） --------------------------
# 用张量存储训练历史（高效且贴合PyTorch逻辑）
w_history = torch.zeros(NUM_EPOCHS)  # 权重变化记录
b_history = torch.zeros(NUM_EPOCHS)  # 偏置变化记录
loss_history = torch.zeros(NUM_EPOCHS)  # 损失变化记录

print("\n开始训练...")
for epoch in range(NUM_EPOCHS):
    # 1. 前向传播：计算预测值
    y_pred = model(X)
    # 2. 计算损失
    loss = criterion(y_pred, y)

    # 3. 记录当前参数和损失
    w_history[epoch] = model.linear.weight.item()
    b_history[epoch] = model.linear.bias.item()
    loss_history[epoch] = loss.item()

    # 4. 反向传播+参数更新
    optimizer.zero_grad()  # 清空上一轮梯度
    loss.backward()  # 自动计算梯度
    optimizer.step()  # 更新权重和偏置

    # 5. 每10轮打印一次训练信息
    if (epoch + 1) % 10 == 0:
        print(
            f"轮次 [{epoch + 1:3d}/{NUM_EPOCHS}] | w: {w_history[epoch]:.6f} | b: {b_history[epoch]:.6f} | 损失: {loss_history[epoch]:.8f}")

# -------------------------- 可视化训练结果（正常显示图像） --------------------------
plt.figure(figsize=(15, 10))  # 设置画布大小

# 1. 权重w的变化曲线
plt.subplot(2, 2, 1)
plt.plot(range(1, NUM_EPOCHS + 1), w_history.numpy(), 'b-', linewidth=2)
plt.axhline(INITIAL_WEIGHT, color='r', linestyle='--', label=f'初始w: {INITIAL_WEIGHT}')
plt.title('训练过程中权重w的变化')
plt.xlabel('训练轮次')
plt.ylabel('权重值')
plt.legend()
plt.grid(alpha=0.3)

# 2. 偏置b的变化曲线
plt.subplot(2, 2, 2)
plt.plot(range(1, NUM_EPOCHS + 1), b_history.numpy(), 'g-', linewidth=2)
plt.axhline(INITIAL_BIAS, color='r', linestyle='--', label=f'初始b: {INITIAL_BIAS}')
plt.title('训练过程中偏置b的变化')
plt.xlabel('训练轮次')
plt.ylabel('偏置值')
plt.legend()
plt.grid(alpha=0.3)

# 3. 损失值的变化曲线
plt.subplot(2, 2, 3)
plt.plot(range(1, NUM_EPOCHS + 1), loss_history.numpy(), 'r-', linewidth=2)
plt.title('训练过程中损失值的变化')
plt.xlabel('训练轮次')
plt.ylabel('MSE损失')
plt.grid(alpha=0.3)

# 4. 模型拟合效果（原始数据+拟合直线）
plt.subplot(2, 2, 4)
plt.scatter(X.numpy(), y.numpy(), alpha=0.5, label='归一化后原始数据', s=30)  # 散点图：原始数据
with torch.no_grad():  # 禁用梯度计算（提高效率）
    y_fit = model(X)  # 计算拟合值
plt.plot(X.numpy(), y_fit.numpy(), 'darkred', linewidth=2.5, label='模型拟合直线')  # 拟合直线
plt.title('模型拟合效果')
plt.xlabel('归一化后特征值')
plt.ylabel('归一化后标签值')
plt.legend()
plt.grid(alpha=0.3)

# 调整子图间距，避免重叠
plt.tight_layout()
# 显示图像（阻塞式，确保图像正常弹出）
plt.show()

# -------------------------- 输出最终训练结果 --------------------------
final_w = model.linear.weight.item()
final_b = model.linear.bias.item()
final_loss = loss_history[-1]

print("\n" + "=" * 60)
print("训练完成！最终结果：")
print(f"  模型公式（归一化后）: y = {final_w:.6f}x + {final_b:.6f}")
print(f"  最终MSE损失: {final_loss:.8f}")
print("=" * 60)