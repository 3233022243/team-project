import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 加入参数b的预测函数：y = w*x + b
def forward(x, w, b):
    return w * x + b

# 加入参数b的损失函数
def loss(x, y, w, b):
    y_pred = forward(x, w, b)
    return (y_pred - y) ** 2

# 读取数据并处理缺失值
try:
    data = pd.read_csv('D:/潘 +深度学习/train.csv')
    data.fillna(data.mean(), inplace=True)
    x_data = data['x'].values
    y_data = data['y'].values
except FileNotFoundError:
    print("错误：未找到train.csv文件，请检查文件路径是否正确")
    exit()
except KeyError as e:
    print(f"错误：数据中缺少必要的列 {e}")
    exit()

# 计算w与损失的关系（固定b）
fixed_b = 0.0
w_list = []
mse_w_list = []
w_range = np.arange(0.0, 4.1, 0.1)
for w in w_range:
    total_loss = 0
    for x_val, y_val in zip(x_data, y_data):
        total_loss += loss(x_val, y_val, w, fixed_b)
    mse = total_loss / len(x_data)
    w_list.append(w)
    mse_w_list.append(mse)
min_index_w = np.argmin(mse_w_list)
best_w = w_list[min_index_w]
min_mse_w = mse_w_list[min_index_w]

# 计算b与损失的关系（固定w）
fixed_w = best_w
b_list = []
mse_b_list = []
b_range = np.arange(-2.0, 2.1, 0.1)
for b in b_range:
    total_loss = 0
    for x_val, y_val in zip(x_data, y_data):
        total_loss += loss(x_val, y_val, fixed_w, b)
    mse = total_loss / len(x_data)
    b_list.append(b)
    mse_b_list.append(mse)
min_index_b = np.argmin(mse_b_list)
best_b = b_list[min_index_b]
min_mse_b = mse_b_list[min_index_b]

# 绘制两个图像
plt.figure(figsize=(14, 6))

# 左图：w与损失的关系
plt.subplot(1, 2, 1)
plt.plot(w_list, mse_w_list, linewidth=2)
plt.scatter(best_w, min_mse_w, c='black', s=100, marker='.', label=f'optimal w: {best_w:.1f}')
plt.xlabel('w')
plt.ylabel('Loss')
plt.title('Relationship between w and Loss')
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()

# 右图：b与损失的关系
plt.subplot(1, 2, 2)
plt.plot(b_list, mse_b_list, linewidth=2)
plt.scatter(best_b, min_mse_b, c='black', s=100, marker='.', label=f'optimal b: {best_b:.1f}')
plt.xlabel('b')
plt.ylabel('Loss')
plt.title('Relationship between b and Loss')
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()

plt.tight_layout()
plt.show()

# 输出最优参数
print(f"\n最优参数: w={best_w:.2f}, b={best_b:.2f}")
print(f"最小损失值: {min_mse_b:.4f}")