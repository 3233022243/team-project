import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# 基础配置
plt.rcParams["font.family"] = ["SimHei", "sans-serif"]
plt.rcParams['axes.unicode_minus'] = False


# 数据加载与预处理
def load_data(file='train.csv'):
    if not os.path.exists(file):
        # 生成示例数据
        x = np.linspace(1, 10, 50).reshape(-1, 1)
        y = 0.5 * x + 2 + np.random.normal(0, 0.1, 50).reshape(-1, 1)
        pd.DataFrame(np.hstack([x, y]), columns=['x', 'y']).to_csv(file, index=False)

    # 读取并预处理数据
    data = pd.read_csv(file).dropna()
    x = data.iloc[:, 0].values.reshape(-1, 1)
    y = data.iloc[:, 1].values.reshape(-1, 1)
    x_norm = (x - x.min()) / (x.max() - x.min() + 1e-6)
    y_norm = (y - y.min()) / (y.max() - y.min() + 1e-6)

    return (torch.tensor(x_norm, dtype=torch.float32),
            torch.tensor(y_norm, dtype=torch.float32),
            (x.min(), x.max(), y.min(), y.max()))


# 线性模型（正态分布初始化）
class LinearModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 1)
        # 正态分布初始化参数
        nn.init.normal_(self.linear.weight, 0, 0.01)
        nn.init.normal_(self.linear.bias, 0, 0.01)

    def forward(self, x):
        return self.linear(x)


# 训练函数
def train(model, x, y, opt, epochs=1000):
    criterion = nn.MSELoss()
    loss_hist, w_hist, b_hist = [], [], []
    for _ in range(epochs):
        y_pred = model(x)
        loss = criterion(y_pred, y)
        # 记录历史
        loss_hist.append(loss.item())
        w_hist.append(model.linear.weight.item())
        b_hist.append(model.linear.bias.item())
        # 反向传播
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        opt.step()
    return loss_hist, w_hist, b_hist


# 主函数
def main():
    x, y, (x_min, x_max, y_min, y_max) = load_data()
    epochs, base_lr = 1000, 0.0005

    # 三种优化器（分别配置参数）
    optimizers = {
        'SGD': lambda params: optim.SGD(params, lr=base_lr, momentum=0.9),
        'Adam': lambda params: optim.Adam(params, lr=base_lr),
        'Adagrad': lambda params: optim.Adagrad(params, lr=base_lr)
    }
    opt_results = {}

    # 训练每种优化器（不包含动画）
    for name, opt_fn in optimizers.items():
        print(f'训练 {name}...')
        model = LinearModel()
        opt = opt_fn(model.parameters())
        loss, w, b = train(model, x, y, opt, epochs)
        opt_results[name] = {'loss': loss, 'w': w, 'b': b, 'model': model}

    # 优化器性能可视化
    plt.figure(figsize=(15, 4))
    # 损失对比
    plt.subplot(131)
    for name, res in opt_results.items():
        plt.plot(res['loss'], label=name)
    plt.xlabel('Epoch'), plt.ylabel('Loss'), plt.title('优化器损失对比'), plt.legend()
    # 权重w变化
    plt.subplot(132)
    for name, res in opt_results.items():
        plt.plot(res['w'], label=name)
    plt.xlabel('Epoch'), plt.ylabel('w'), plt.title('权重w调节过程'), plt.legend()
    # 偏置b变化
    plt.subplot(133)
    for name, res in opt_results.items():
        plt.plot(res['b'], label=name)
    plt.xlabel('Epoch'), plt.ylabel('b'), plt.title('偏置b调节过程'), plt.legend()
    plt.tight_layout(), plt.show()

    # 不同epoch对比
    print('测试不同epoch...')
    plt.figure(figsize=(12, 5))
    for ep in [200, 500, 1000, 2000]:
        model = LinearModel()
        loss, _, _ = train(model, x, y, optim.Adam(model.parameters(), lr=base_lr), ep)
        plt.plot(loss, label=f'epoch={ep}')
    plt.xlabel('Epoch'), plt.ylabel('Loss'), plt.title('不同epoch调节过程'), plt.legend(), plt.show()

    # 不同学习率对比
    print('测试不同学习率...')
    plt.figure(figsize=(12, 5))
    for lr in [0.1, 0.01, 0.005, 0.001]:
        model = LinearModel()
        loss, _, _ = train(model, x, y, optim.Adam(model.parameters(), lr=lr), epochs)
        plt.plot(loss, label=f'lr={lr}')
    plt.xlabel('Epoch'), plt.ylabel('Loss'), plt.title('不同学习率调节过程'), plt.legend(), plt.show()

    # 输出最终参数
    print('\n==== 最终参数（原始尺度）====')
    x_range, y_range = x_max - x_min, y_max - y_min
    for name, res in opt_results.items():
        w_norm = res['model'].linear.weight.item()
        b_norm = res['model'].linear.bias.item()
        w_raw = w_norm * y_range / x_range
        b_raw = b_norm * y_range + y_min - w_raw * x_min
        print(f'{name}: w={w_raw:.4f}, b={b_raw:.4f}, 最终损失={res["loss"][-1]:.6f}')


if __name__ == "__main__":
    main()