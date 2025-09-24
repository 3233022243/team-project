import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# 设置中文显示
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
plt.rcParams["axes.unicode_minus"] = False


class LinearRegression:
    """简化版线性回归模型 y = wx + b"""
    
    def __init__(self, learning_rate=0.01, epochs=1000):
        self.w = np.random.randn()  # 随机初始化权重
        self.b = np.random.randn()  # 随机初始化偏置
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.loss_history = []  # 损失历史
        self.w_history = []     # 权重历史
        self.b_history = []     # 偏置历史

    def compute_loss(self, x, y):
        """计算均方误差损失"""
        y_pred = self.w * x + self.b
        return np.mean((y_pred - y) ** 2)

    def gradient_descent(self, x, y):
        """梯度下降更新参数"""
        y_pred = self.w * x + self.b
        dw = 2 * np.mean((y_pred - y) * x)
        db = 2 * np.mean(y_pred - y)
        
        # 更新参数
        self.w -= self.learning_rate * dw
        self.b -= self.learning_rate * db

    def train(self, x, y):
        """训练模型"""
        # 标准化数据
        x_mean, x_std = np.mean(x), np.std(x)
        y_mean, y_std = np.mean(y), np.std(y)
        x_norm = (x - x_mean) / x_std
        y_norm = (y - y_mean) / y_std

        # 训练过程
        for epoch in range(self.epochs):
            self.loss_history.append(self.compute_loss(x_norm, y_norm))
            self.w_history.append(self.w)
            self.b_history.append(self.b)
            self.gradient_descent(x_norm, y_norm)
            
            # 每200轮打印一次信息
            if (epoch + 1) % 200 == 0:
                print(f"轮次 {epoch + 1}/{self.epochs}, 损失: {self.loss_history[-1]:.6f}")

        # 转换回原始尺度的参数
        self.w_original = self.w * (y_std / x_std)
        self.b_original = (self.b * y_std) + y_mean - (self.w_original * x_mean)

        return self.w_original, self.b_original


def main():
    # 1. 读取数据
    csv_path = input("请输入CSV文件路径（默认为train.csv）: ") or "train.csv"

    if not os.path.exists(csv_path):
        print(f"错误：未找到文件 {csv_path}")
        return

    try:
        df = pd.read_csv(csv_path)
        print(f"成功读取数据：{csv_path}，共 {len(df)} 条记录")
        
        # 检查必要列
        if not all(col in df.columns for col in ['x', 'y']):
            print(f"错误：CSV需包含'x'和'y'列，当前列名：{df.columns.tolist()}")
            return

        # 提取数据并去除缺失值
        df = df[['x', 'y']].dropna()
        x, y = df['x'].values, df['y'].values

    except Exception as e:
        print(f"读取数据失败：{str(e)}")
        return

    # 2. 训练模型
    model = LinearRegression(epochs=1000)
    print("\n开始训练模型...")
    w, b = model.train(x, y)
    print(f"\n训练完成！拟合公式: y = {w:.4f}x + {b:.4f}")

    # 3. 绘制结果
    plt.figure(figsize=(12, 10))

    # 数据点与拟合直线
    plt.subplot(2, 2, 1)
    plt.scatter(x, y, color='skyblue', label='原始数据', alpha=0.6)
    plt.plot(x, w * x + b, color='crimson', linewidth=2, label=f'y = {w:.2f}x + {b:.2f}')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('数据分布与线性拟合')
    plt.legend()

    # w与loss关系
    plt.subplot(2, 2, 2)
    plt.plot(model.w_history, model.loss_history, color='forestgreen')
    plt.xlabel('权重 w')
    plt.ylabel('损失值')
    plt.title('w 与损失的关系')

    # b与loss关系
    plt.subplot(2, 2, 3)
    plt.plot(model.b_history, model.loss_history, color='darkorchid')
    plt.xlabel('偏置 b')
    plt.ylabel('损失值')
    plt.title('b 与损失的关系')

    # 损失随轮次变化
    plt.subplot(2, 2, 4)
    plt.plot(range(len(model.loss_history)), model.loss_history, color='darkorange')
    plt.xlabel('训练轮次')
    plt.ylabel('损失值')
    plt.title('损失值下降趋势')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
