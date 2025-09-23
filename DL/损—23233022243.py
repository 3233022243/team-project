import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D

# 解决中文显示问题
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题

# 读取train.csv文件（使用指定的绝对路径）
file_path = "C:/Users/dell/Downloads/train.csv"  # Windows路径使用正斜杠

try:
    # 读取CSV文件
    df = pd.read_csv(file_path)
    print(f"成功读取数据：共 {len(df)} 条记录")

    # 提取x和y数据（假设第一列是x，第二列是y）
    # 如果实际列名不同，请修改为列名，例如df['x_column'].values
    x_data = df.iloc[:, 0].values
    y_data = df.iloc[:, 1].values

except FileNotFoundError:
    print(f"错误：未找到文件，请检查路径是否正确 -> {file_path}")
    exit()
except Exception as e:
    print(f"读取数据时发生错误：{str(e)}")
    exit()


# 线性模型：y_pred = w*x + b
def forward(x, w, b):
    return w * x + b


# 损失函数（均方误差）
def loss(x, y, w, b):
    y_pred = forward(x, w, b)
    return (y_pred - y) ** 2


# 设置参数搜索范围（可根据数据分布调整）
w_min, w_max = -5.0, 15.0  # 扩大权重w的范围以适应更多数据情况
b_min, b_max = -10.0, 10.0  # 偏置b的范围
w_range = np.arange(w_min, w_max + 0.1, 0.5)  # 步长0.5
b_range = np.arange(b_min, b_max + 0.1, 0.5)
W, B = np.meshgrid(w_range, b_range)  # 生成参数网格


# 优化损失计算（使用向量化操作提高效率）
def compute_loss(w, b):
    y_pred = forward(x_data, w, b)
    return np.mean((y_pred - y_data) ** 2)


# 向量化计算所有损失值
loss_values = np.vectorize(compute_loss)(W, B)

# 绘制3D损失曲面
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

# 绘制曲面
surf = ax.plot_surface(W, B, loss_values,
                       cmap='viridis',  # 颜色映射
                       alpha=0.8,  # 透明度
                       edgecolor='none')

# 设置坐标轴和标题
ax.set_xlabel('w (权重)', fontsize=10)
ax.set_ylabel('b (偏置)', fontsize=10)
ax.set_zlabel('损失值 (MSE)', fontsize=10)
ax.set_title('线性模型(wx + b)的损失函数曲面', fontsize=12)

# 添加颜色条（表示损失值大小）
fig.colorbar(surf, ax=ax, shrink=0.5, aspect=8)

# 显示图像
plt.show()
