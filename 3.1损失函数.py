import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# 读取数据集
df = pd.read_csv('train.csv')

# 检查数据中的缺失值
print('数据中的缺失值：')
print(df.isnull().sum())

# 检查数据中的无穷大值
print('数据中的无穷大值：')
print(df[df.isin([np.inf, -np.inf]).any(axis=1)])

# 处理缺失值，这里采用删除缺失值所在行
df = df.dropna()

# 处理无穷大值，这里采用删除无穷大值所在行
df = df.replace([np.inf, -np.inf], np.nan).dropna()

# 提取特征和目标变量
x = df['x'].values
y = df['y'].values

# 初始化参数
w = 0
b = 0

# 学习率和迭代次数
learning_rate = 0.000001
epochs = 100

# 存储损失、w 和 b 的值
losses = []
ws = []
bs = []

# 梯度下降
for i in range(epochs):
    y_pred = w * x + b
    loss = np.mean((y_pred - y) ** 2)
    dw = np.mean(2 * (y_pred - y) * x)
    db = np.mean(2 * (y_pred - y))
    w = w - learning_rate * dw
    b = b - learning_rate * db

    losses.append(loss)
    ws.append(w)
    bs.append(b)

# 设置图片清晰度
plt.rcParams['figure.dpi'] = 300

# 设置 matplotlib 支持中文
plt.rcParams['font.sans-serif'] = ['WenQuanYi Zen Hei']
plt.rcParams['axes.unicode_minus'] = False

# 绘制 w 和损失之间的关系
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(ws, losses)
plt.xlabel('w')
plt.xticks(rotation=45)
plt.ylabel('损失')
plt.title('w 和损失之间的关系')

# 绘制 b 和损失之间的关系
plt.subplot(1, 2, 2)
plt.plot(bs, losses)
plt.xlabel('b')
plt.xticks(rotation=45)
plt.ylabel('损失')
plt.title('b 和损失之间的关系')

plt.tight_layout()
plt.show()

print(f'训练后的 w: {w}')
print(f'训练后的 b: {b}')