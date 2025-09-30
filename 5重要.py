import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset, DataLoader
import seaborn as sns
import os
import time
from matplotlib.ticker import MaxNLocator

# ----------------------------
# 1. 配置与初始化（修复字体问题）
# ----------------------------
FILE_PATH = r"C:\Users\dell\Desktop\train.csv"
BASE_LR = 0.02
EPOCHS = 300
BATCH_SIZE = 16

# 可视化风格配置
try:
    plt.style.use('seaborn-talk')
except:
    plt.style.use('default')

# 字体配置 - 解决负号显示问题
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC", "Arial Unicode MS", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False  # 确保负号正确显示
plt.rcParams["axes.labelpad"] = 10
plt.rcParams["axes.titlepad"] = 15
plt.rcParams["legend.fontsize"] = 11
plt.rcParams["xtick.labelsize"] = 10
plt.rcParams["ytick.labelsize"] = 10
plt.rcParams["figure.dpi"] = 150

# 专业配色方案
COLORS = {
    'primary': '#2c7fb8',
    'secondary': '#e1812c',
    'tertiary': '#31a354',
    'quaternary': '#9e9ac8',
    'accent': '#d95f0e',
    'light': '#f7f7f7',
    'dark': '#333333'
}


# ----------------------------
# 2. 数据加载与增强处理
# ----------------------------
def load_and_preprocess_data():
    if not os.path.exists(FILE_PATH):
        raise FileNotFoundError(f"数据集不存在，请检查路径: {FILE_PATH}")

    df = pd.read_csv(FILE_PATH)
    print(f"原始数据形状: {df.shape}")

    # 数据验证与转换
    if 'x' not in df.columns or 'y' not in df.columns:
        raise ValueError("数据集必须包含'x'和'y'列")
    df['x'] = pd.to_numeric(df['x'], errors='coerce')
    df['y'] = pd.to_numeric(df['y'], errors='coerce')

    # 缺失值处理
    print("\n缺失值统计:")
    missing = df.isnull().sum()
    print(missing)

    if missing['x'] > 0 or missing['y'] > 0:
        df['x'] = df['x'].interpolate(method='linear').fillna(df['x'].median())
        df['y'] = df['y'].interpolate(method='linear').fillna(df['y'].median())

    # 异常值处理
    def handle_outliers(col):
        q1, q3 = df[col].quantile([0.25, 0.75])
        iqr = q3 - q1
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        outliers = len(df[(df[col] < lower) | (df[col] > upper)])
        df[col] = df[col].clip(lower, upper)
        print(f"{col}列处理了{outliers}个异常值")
        return df

    df = handle_outliers('x')
    df = handle_outliers('y')

    # 数据清洗前后对比可视化
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    raw_data = pd.read_csv(FILE_PATH)

    # 原始数据散点图
    scatter1 = ax1.scatter(raw_data['x'], raw_data['y'], alpha=0.6,
                           color=COLORS['primary'], edgecolor='white', s=50)
    ax1.set_title('清洗前数据分布', fontsize=14)
    ax1.set_xlabel('x 特征', fontsize=12)
    ax1.set_ylabel('y 标签', fontsize=12)
    ax1.grid(color='gray', linestyle='--', alpha=0.3)
    ax1.set_facecolor(COLORS['light'])

    # 清洗后数据散点图
    scatter2 = ax2.scatter(df['x'], df['y'], alpha=0.6,
                           color=COLORS['secondary'], edgecolor='white', s=50)
    ax2.set_title('清洗后数据分布', fontsize=14)
    ax2.set_xlabel('x 特征', fontsize=12)
    ax2.set_ylabel('y 标签', fontsize=12)
    ax2.grid(color='gray', linestyle='--', alpha=0.3)
    ax2.set_facecolor(COLORS['light'])

    plt.gcf().set_facecolor(COLORS['light'])
    plt.tight_layout()
    plt.savefig('data_cleaning_comparison.png', bbox_inches='tight')
    plt.show()

    # 数据标准化与准备
    scaler_x = StandardScaler()
    scaler_y = StandardScaler()
    x_scaled = scaler_x.fit_transform(df['x'].values.reshape(-1, 1))
    y_scaled = scaler_y.fit_transform(df['y'].values.reshape(-1, 1))

    dataset = TensorDataset(
        torch.tensor(x_scaled, dtype=torch.float32),
        torch.tensor(y_scaled, dtype=torch.float32)
    )
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    return dataloader, scaler_x, scaler_y, df


# ----------------------------
# 3. 模型定义与初始化可视化
# ----------------------------
class EnhancedLinearModel(nn.Module):
    def __init__(self, mean=0.0, std=0.3):
        super(EnhancedLinearModel, self).__init__()
        self.linear = nn.Linear(1, 1)
        nn.init.normal_(self.linear.weight, mean=mean, std=std)
        nn.init.normal_(self.linear.bias, mean=mean, std=std)

    def forward(self, x):
        return self.linear(x)


# 初始化分布可视化
def verify_initialization():
    weights = []
    biases = []
    for _ in range(1000):
        model = EnhancedLinearModel()
        weights.append(model.linear.weight.item())
        biases.append(model.linear.bias.item())

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # 权重分布
    sns.kdeplot(weights, ax=ax1, fill=True, color=COLORS['primary'], linewidth=2)
    ax1.set_title('权重初始化分布 (μ=0, σ=0.3)', fontsize=14)
    ax1.axvline(x=0, color=COLORS['accent'], linestyle='--', linewidth=2)
    ax1.set_xlabel('权重值', fontsize=12)
    ax1.set_ylabel('概率密度', fontsize=12)
    ax1.grid(color='gray', linestyle='--', alpha=0.3)
    ax1.set_facecolor(COLORS['light'])

    # 偏置分布
    sns.kdeplot(biases, ax=ax2, fill=True, color=COLORS['tertiary'], linewidth=2)
    ax2.set_title('偏置初始化分布 (μ=0, σ=0.3)', fontsize=14)
    ax2.axvline(x=0, color=COLORS['accent'], linestyle='--', linewidth=2)
    ax2.set_xlabel('偏置值', fontsize=12)
    ax2.set_ylabel('概率密度', fontsize=12)
    ax2.grid(color='gray', linestyle='--', alpha=0.3)
    ax2.set_facecolor(COLORS['light'])

    plt.gcf().set_facecolor(COLORS['light'])
    plt.tight_layout()
    plt.savefig('parameter_initialization.png', bbox_inches='tight')
    plt.show()


# ----------------------------
# 4. 训练函数与日志记录
# ----------------------------
def train_model(model, criterion, optimizer, scheduler, dataloader, epochs=300):
    log = {
        'loss': [],
        'w': [],
        'b': [],
        'lr': [],
        'epochs': list(range(epochs))
    }

    model.train()
    start_time = time.time()

    for epoch in range(epochs):
        total_loss = 0.0
        for x_batch, y_batch in dataloader:
            y_pred = model(x_batch)
            loss = criterion(y_pred, y_batch)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        log['loss'].append(avg_loss)
        log['w'].append(model.linear.weight.item())
        log['b'].append(model.linear.bias.item())
        log['lr'].append(optimizer.param_groups[0]['lr'])
        scheduler.step()

        if (epoch + 1) % 50 == 0:
            print(f"Epoch [{epoch + 1}/{epochs}] - 损失: {avg_loss:.6f}, "
                  f"w: {model.linear.weight.item():.6f}, b: {model.linear.bias.item():.6f}")

    print(f"训练耗时: {time.time() - start_time:.2f}秒")
    return log


# ----------------------------
# 5. 优化器对比实验
# ----------------------------
def compare_optimizers(dataloader):
    criterion = nn.MSELoss()

    optimizers = {
        'Adagrad': torch.optim.Adagrad(EnhancedLinearModel().parameters(), lr=BASE_LR),
        'Adam': torch.optim.Adam(EnhancedLinearModel().parameters(), lr=BASE_LR),
        'Adamax': torch.optim.Adamax(EnhancedLinearModel().parameters(), lr=BASE_LR)
    }

    schedulers = {
        name: torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
        for name, opt in optimizers.items()
    }

    logs = {}
    print("\n===== 优化器对比实验 =====")
    for name, opt in optimizers.items():
        print(f"\n----- 训练 {name} 优化器 -----")
        model = EnhancedLinearModel()
        logs[name] = train_model(model, criterion, opt, schedulers[name], dataloader, EPOCHS)

    # 优化器对比可视化
    plt.figure(figsize=(16, 8))
    styles = ['-', '--', '-.']
    markers = ['o', 's', '^']
    for i, (name, log) in enumerate(logs.items()):
        plt.plot(log['epochs'], log['loss'], label=f'{name} 优化器',
                 color=list(COLORS.values())[i + 1], linewidth=2,
                 linestyle=styles[i], markersize=5, markevery=20)

    plt.title('不同优化器的损失曲线对比', fontsize=16)
    plt.xlabel('迭代次数', fontsize=14)
    plt.ylabel('MSE 损失', fontsize=14)
    plt.yscale('log')
    plt.grid(color='gray', linestyle='--', alpha=0.3)
    plt.legend(fontsize=12)
    plt.gcf().set_facecolor(COLORS['light'])
    plt.tight_layout()
    plt.savefig('optimizer_comparison.png', bbox_inches='tight')
    plt.show()

    return logs


# ----------------------------
# 6. 参数调节过程可视化
# ----------------------------
def visualize_parameter_adjustment(logs):
    best_name = min(logs.items(), key=lambda x: x[1]['loss'][-1])[0]
    best_log = logs[best_name]
    print(f"\n最佳优化器: {best_name}，最终损失: {best_log['loss'][-1]:.6f}")

    # 创建2行1列的图表
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12), sharex=True)

    # 权重调节曲线
    ax1.plot(best_log['epochs'], best_log['w'], color=COLORS['primary'],
             linewidth=2, alpha=0.8)
    ax1.axhline(y=best_log['w'][-1], color=COLORS['accent'],
                linestyle='--', linewidth=2,
                label=f'最终权重: {best_log["w"][-1]:.6f}')
    ax1.set_title(f'权重(w)调节过程 - {best_name}优化器', fontsize=14)
    ax1.set_ylabel('权重值', fontsize=12)
    ax1.legend(fontsize=12)
    ax1.grid(color='gray', linestyle='--', alpha=0.3)
    ax1.set_facecolor(COLORS['light'])

    # 偏置调节曲线
    ax2.plot(best_log['epochs'], best_log['b'], color=COLORS['tertiary'],
             linewidth=2, alpha=0.8)
    ax2.axhline(y=best_log['b'][-1], color=COLORS['accent'],
                linestyle='--', linewidth=2,
                label=f'最终偏置: {best_log["b"][-1]:.6f}')
    ax2.set_title(f'偏置(b)调节过程 - {best_name}优化器', fontsize=14)
    ax2.set_xlabel('迭代次数', fontsize=12)
    ax2.set_ylabel('偏置值', fontsize=12)
    ax2.legend(fontsize=12)
    ax2.grid(color='gray', linestyle='--', alpha=0.3)
    ax2.set_facecolor(COLORS['light'])
    ax2.xaxis.set_major_locator(MaxNLocator(integer=True))

    plt.gcf().set_facecolor(COLORS['light'])
    plt.tight_layout()
    plt.savefig('parameter_adjustment.png', bbox_inches='tight')
    plt.show()

    return best_name, best_log


# ----------------------------
# 7. 超参数调节可视化
# ----------------------------
def visualize_hyperparameters(dataloader):
    criterion = nn.MSELoss()

    # 学习率调节实验
    print("\n===== 学习率调节实验 =====")
    lr_values = [0.005, 0.01, 0.02, 0.05, 0.1]
    lr_logs = {}

    for lr in lr_values:
        print(f"测试学习率: {lr}")
        model = EnhancedLinearModel()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
        lr_logs[lr] = train_model(model, criterion, optimizer, scheduler, dataloader, epochs=200)

    # 学习率可视化
    plt.figure(figsize=(16, 8))
    for i, (lr, log) in enumerate(lr_logs.items()):
        plt.plot(log['epochs'], log['loss'], label=f'学习率 = {lr}',
                 color=plt.cm.viridis(i / len(lr_values)), linewidth=2,
                 markevery=25, markersize=6)

    plt.title('不同学习率对训练损失的影响', fontsize=16)
    plt.xlabel('迭代次数', fontsize=14)
    plt.ylabel('MSE 损失', fontsize=14)
    plt.yscale('log')
    plt.grid(color='gray', linestyle='--', alpha=0.3)
    plt.legend(title='学习率', title_fontsize=12, fontsize=11)
    plt.gcf().set_facecolor(COLORS['light'])
    plt.tight_layout()
    plt.savefig('learning_rate_impact.png', bbox_inches='tight')
    plt.show()

    # 迭代次数调节实验
    print("\n===== 迭代次数调节实验 =====")
    epoch_values = [50, 100, 200, 300, 400]
    epoch_logs = {}

    for epochs in epoch_values:
        print(f"测试迭代次数: {epochs}")
        model = EnhancedLinearModel()
        optimizer = torch.optim.Adam(model.parameters(), lr=BASE_LR)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        epoch_logs[epochs] = train_model(model, criterion, optimizer, scheduler, dataloader, epochs=epochs)

    # 迭代次数可视化
    final_losses = {e: log['loss'][-1] for e, log in epoch_logs.items()}
    fig, ax = plt.subplots(figsize=(16, 8))

    bars = ax.bar(final_losses.keys(), final_losses.values(),
                  color=COLORS['secondary'], edgecolor='black', alpha=0.7)

    # 添加数值标签
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., height,
                f'{height:.6f}', ha='center', va='bottom', fontsize=10)

    ax.set_title('最终损失与迭代次数的关系', fontsize=16)
    ax.set_xlabel('迭代次数', fontsize=14)
    ax.set_ylabel('最终 MSE 损失', fontsize=14)
    ax.grid(axis='y', color='gray', linestyle='--', alpha=0.3)
    ax.set_facecolor(COLORS['light'])
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    plt.gcf().set_facecolor(COLORS['light'])
    plt.tight_layout()
    plt.savefig('epoch_impact.png', bbox_inches='tight')
    plt.show()

    return lr_logs, epoch_logs


# ----------------------------
# 8. 保存最佳模型
# ----------------------------
def save_best_model(best_name, best_log):
    model = EnhancedLinearModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=BASE_LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    dataloader, _, _, _ = load_and_preprocess_data()
    train_model(model, nn.MSELoss(), optimizer, scheduler, dataloader, EPOCHS)

    torch.save({
        'model_state_dict': model.state_dict(),
        'final_loss': best_log['loss'][-1],
        'best_optimizer': best_name,
        'w': best_log['w'][-1],
        'b': best_log['b'][-1]
    }, 'best_model.pth')
    print("\n最佳模型已保存为 best_model.pth")


# ----------------------------
# 主函数
# ----------------------------
def main():
    dataloader, _, _, _ = load_and_preprocess_data()
    verify_initialization()
    optimizer_logs = compare_optimizers(dataloader)
    best_name, best_log = visualize_parameter_adjustment(optimizer_logs)
    visualize_hyperparameters(dataloader)
    save_best_model(best_name, best_log)
    print("\n所有任务完成！生成的可视化图表已优化至最佳效果")


if __name__ == "__main__":
    main()
