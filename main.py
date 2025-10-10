import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import matplotlib

# 更换为Agg后端，避免GUI相关问题
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os
from datetime import datetime
import io
from PIL import Image

# 设置中文显示
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题


# 图像处理辅助函数 - 避免直接使用matplotlib的tostring相关方法
def save_fig_safely(fig, filename, dpi=300):
    """安全保存图像，通过PIL中转处理"""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', dpi=dpi)
    buf.seek(0)
    # 用PIL打开并保存，避免直接操作matplotlib的底层对象
    img = Image.open(buf)
    img.save(filename)
    return img


# 1. 数据加载与预处理
def load_data(file_path='train.csv'):
    """加载并预处理数据"""
    if not os.path.exists(file_path):
        # 若文件不存在，生成模拟数据避免报错
        np.random.seed(42)
        X = np.random.randn(100, 5)
        y = 2 * X[:, 0] + 3 * X[:, 1] - X[:, 2] + np.random.randn(100)
        df = pd.DataFrame(np.column_stack((X, y)))
    else:
        df = pd.read_csv(file_path)

    # 假设最后一列是目标变量
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values

    # 数据分割
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 数据标准化 - 分开处理特征和目标变量
    scaler_X = StandardScaler()
    X_train = scaler_X.fit_transform(X_train)
    X_test = scaler_X.transform(X_test)

    scaler_y = StandardScaler()
    y_train = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()
    y_test = scaler_y.transform(y_test.reshape(-1, 1)).flatten()

    # 转换为Tensor
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32)

    return X_train, X_test, y_train, y_test


# 2. 定义模型
class LinearModel(nn.Module):
    def __init__(self, input_dim):
        super(LinearModel, self).__init__()
        self.linear = nn.Linear(input_dim, 1)

        # 初始化权重和偏置为正态分布
        nn.init.normal_(self.linear.weight, mean=0.0, std=0.01)
        nn.init.normal_(self.linear.bias, mean=0.0, std=0.01)

    def forward(self, x):
        return self.linear(x).flatten()

    def get_parameters(self):
        """获取当前权重和偏置"""
        w = self.linear.weight.detach().numpy().flatten()
        b = self.linear.bias.detach().numpy().item()
        return w, b


# 3. 训练函数
def train_model(model, optimizer, criterion, X_train, y_train, X_test, y_test,
                epochs=100, track_params=False):
    """训练模型并记录训练过程"""
    train_losses = []
    test_losses = []
    w_history = []
    b_history = []

    for epoch in range(epochs):
        # 训练模式
        model.train()
        optimizer.zero_grad()

        # 前向传播
        outputs = model(X_train)
        loss = criterion(outputs, y_train)

        # 反向传播和优化
        loss.backward()
        optimizer.step()

        # 记录训练损失
        train_losses.append(loss.item())

        # 在测试集上评估
        model.eval()
        with torch.no_grad():
            test_outputs = model(X_test)
            test_loss = criterion(test_outputs, y_test)
            test_losses.append(test_loss.item())

        # 跟踪参数变化
        if track_params:
            w, b = model.get_parameters()
            w_history.append(w)
            b_history.append(b)

        # 每10%的epoch打印一次信息
        if (epoch + 1) % (epochs // 10) == 0:
            print(f'Epoch [{epoch + 1}/{epochs}], Train Loss: {loss.item():.4f}, Test Loss: {test_loss.item():.4f}')

    return {
        'train_losses': train_losses,
        'test_losses': test_losses,
        'w_history': np.array(w_history) if track_params else None,
        'b_history': np.array(b_history) if track_params else None,
        'model': model
    }


# 4. 可视化函数
def plot_optimizer_comparison(results, optimizer_names, metric='test'):
    """比较不同优化器的性能"""
    fig, ax = plt.subplots(figsize=(12, 6))

    for i, name in enumerate(optimizer_names):
        if metric == 'test':
            ax.plot(results[i]['test_losses'], label=name)
        else:
            ax.plot(results[i]['train_losses'], label=name)

    ax.set_title(f'不同优化器的{metric}损失对比')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('损失值')
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()

    # 使用安全保存函数
    save_fig_safely(fig, f'optimizer_comparison_{metric}.png')
    plt.close(fig)  # 关闭图像释放资源


def plot_parameter_changes(w_history, b_history, param_name='权重w'):
    """可视化参数变化过程"""
    fig, ax = plt.subplots(figsize=(12, 6))

    if param_name == '权重w' and w_history is not None and w_history.size > 0:
        # 如果有多个权重，分别绘制
        num_weights = w_history.shape[1]
        for i in range(num_weights):
            ax.plot(w_history[:, i], label=f'w{i}')
        ax.set_title('权重参数w的变化过程')
    elif param_name == '偏置b' and b_history is not None and len(b_history) > 0:
        ax.plot(b_history, label='b')
        ax.set_title('偏置参数b的变化过程')
    else:
        ax.text(0.5, 0.5, '无有效参数历史可绘制', ha='center', va='center', fontsize=12)

    ax.set_xlabel('Epoch')
    ax.set_ylabel('参数值')
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()

    # 使用安全保存函数
    save_fig_safely(fig, f'{param_name}_changes.png')
    plt.close(fig)  # 关闭图像释放资源


def plot_hyperparameter_effect(results, param_values, param_name='学习率'):
    """可视化超参数调节效果"""
    fig, ax = plt.subplots(figsize=(12, 6))

    for i, value in enumerate(param_values):
        if len(results[i]['test_losses']) > 0:
            ax.plot(results[i]['test_losses'], label=f'{param_name}={value}')

    ax.set_title(f'不同{param_name}对测试损失的影响')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('测试损失值')
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()

    # 使用安全保存函数
    save_fig_safely(fig, f'{param_name}_effect.png')
    plt.close(fig)  # 关闭图像释放资源


# 5. 主函数
def main():
    # 加载数据
    print("加载数据...")
    try:
        X_train, X_test, y_train, y_test = load_data()
    except Exception as e:
        print(f"数据加载异常: {e}")
        return
    input_dim = X_train.shape[1]
    print(f"数据加载完成，特征维度: {input_dim}")

    # 定义损失函数
    criterion = nn.MSELoss()

    # 选择三种不同的优化器
    optimizer_classes = [
        optim.Adam,  # Adam优化器
        optim.SGD,  # 随机梯度下降
        optim.RMSprop  # RMSprop优化器
    ]
    optimizer_names = ['Adam', 'SGD', 'RMSprop']
    optimizer_results = []

    # 使用不同优化器训练模型
    print("\n使用不同优化器训练模型...")
    for i, optim_class in enumerate(optimizer_classes):
        print(f"\n使用{optimizer_names[i]}优化器:")
        model = LinearModel(input_dim)

        # 设置不同优化器的典型参数
        if optim_class == optim.SGD:
            optimizer = optim_class(model.parameters(), lr=0.01, momentum=0.9)
        else:
            optimizer = optim_class(model.parameters(), lr=0.001)

        result = train_model(
            model, optimizer, criterion,
            X_train, y_train, X_test, y_test,
            epochs=100, track_params=True
        )
        optimizer_results.append(result)

    # 可视化不同优化器的性能
    print("\n绘制优化器性能对比图...")
    plot_optimizer_comparison(optimizer_results, optimizer_names, metric='train')
    plot_optimizer_comparison(optimizer_results, optimizer_names, metric='test')

    # 可视化参数w和b的调节过程（使用性能最好的优化器结果）
    best_idx = np.argmin([min(res['test_losses']) for res in optimizer_results])
    print(f"\n最佳优化器是: {optimizer_names[best_idx]}")

    print("绘制参数w变化图...")
    plot_parameter_changes(
        optimizer_results[best_idx]['w_history'],
        optimizer_results[best_idx]['b_history'],
        param_name='权重w'
    )

    print("绘制参数b变化图...")
    plot_parameter_changes(
        optimizer_results[best_idx]['w_history'],
        optimizer_results[best_idx]['b_history'],
        param_name='偏置b'
    )

    # 调节学习率并可视化
    print("\n调节学习率并可视化...")
    learning_rates = [0.0001, 0.001, 0.01, 0.1]
    lr_results = []

    for lr in learning_rates:
        print(f"使用学习率 {lr}:")
        model = LinearModel(input_dim)
        optimizer = optimizer_classes[best_idx](model.parameters(), lr=lr)

        result = train_model(
            model, optimizer, criterion,
            X_train, y_train, X_test, y_test,
            epochs=100
        )
        lr_results.append(result)

    plot_hyperparameter_effect(lr_results, learning_rates, param_name='学习率')

    # 调节epoch并可视化
    print("\n调节epoch并可视化...")
    epochs_list = [50, 100, 200, 300]
    epoch_results = []

    for epochs in epochs_list:
        print(f"使用epochs {epochs}:")
        model = LinearModel(input_dim)
        optimizer = optimizer_classes[best_idx](model.parameters(), lr=0.001)

        result = train_model(
            model, optimizer, criterion,
            X_train, y_train, X_test, y_test,
            epochs=epochs
        )
        epoch_results.append(result)

    plot_hyperparameter_effect(epoch_results, epochs_list, param_name='epoch')

    # 保存最佳模型
    print("\n保存最佳模型...")
    best_model = optimizer_results[best_idx]['model']
    torch.save(best_model.state_dict(), 'best_model.pth')
    print("最佳模型已保存为 best_model.pth")

    # 提示推送到Git仓库
    print("\n请将代码推送到Git仓库，并在备注中包含您的学号")
    print("推送命令示例:")
    print("git add .")
    print("git commit -m '提交最佳模型代码，学号: [您的学号]'")
    print("git push origin main")


if __name__ == "__main__":
    main()
