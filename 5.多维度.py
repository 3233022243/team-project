import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import warnings
warnings.filterwarnings('ignore')


plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False




# 设置随机种子
torch.manual_seed(42)
np.random.seed(42)


# 1. 数据集定义
class CountriesDataset(Dataset):
    def __init__(self, features, targets):
        self.features = torch.FloatTensor(features)
        self.targets = torch.FloatTensor(targets)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]


# 2. 神经网络模型
class FiveLayerNetwork(nn.Module):
    def __init__(self, input_size):
        super(FiveLayerNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 7),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(7, 6),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(6, 5),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(5, 3),
            nn.ReLU(),
            nn.Linear(3, 1)
        )

    def forward(self, x):
        return self.network(x)


# 3. 数据预处理函数
def preprocess_data(df):
    print("原始数据信息:")
    print(f"数据形状: {df.shape}")
    print(f"列名: {df.columns.tolist()}")
    print(f"\n数据类型:")
    print(df.dtypes)

    # 复制数据避免修改原始数据
    df_processed = df.copy()

    # 处理非数值列
    non_numeric_cols = []
    for col in df_processed.columns:
        if df_processed[col].dtype == 'object':
            non_numeric_cols.append(col)
            print(f"处理非数值列: {col}")
            # 使用标签编码
            le = LabelEncoder()
            df_processed[col] = le.fit_transform(df_processed[col].astype(str))

    if non_numeric_cols:
        print(f"已处理的非数值列: {non_numeric_cols}")

    # 检查并处理缺失值
    missing_values = df_processed.isnull().sum()
    if missing_values.any():
        print(f"\n缺失值情况:")
        print(missing_values[missing_values > 0])
        # 用中位数填充数值列
        for col in df_processed.columns:
            if df_processed[col].isnull().any():
                df_processed[col].fillna(df_processed[col].median(), inplace=True)

    return df_processed


# 4. 加载和预处理数据
def dataloader():
    try:
        df = pd.read_csv('countries.csv')
        print("数据加载成功!")
    except FileNotFoundError:
        print("错误: 找不到 countries.csv 文件")
        print("请确保文件在当前目录下")
        return None, None, None, None

    # 预处理数据
    df_processed = preprocess_data(df)

    # 假设最后一列是目标变量
    # 如果不是，请根据实际情况调整
    X = df_processed.iloc[:, :-1].values
    y = df_processed.iloc[:, -1].values.reshape(-1, 1)

    print(f"\n处理后的特征形状: {X.shape}")
    print(f"目标变量形状: {y.shape}")

    # 数据标准化
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()

    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y)

    return X_scaled, y_scaled, scaler_X, scaler_y


# 5. 训练函数
def train_model(model, train_loader, test_loader, device):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    train_losses = []
    test_losses = []
    best_loss = float('inf')

    print("\n开始训练...")
    for epoch in range(100):
        # 训练
        model.train()
        train_loss = 0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # 测试
        model.eval()
        test_loss = 0
        with torch.no_grad():
            for batch_X, batch_y in test_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                test_loss += loss.item()

        avg_train = train_loss / len(train_loader)
        avg_test = test_loss / len(test_loader)
        train_losses.append(avg_train)
        test_losses.append(avg_test)

        # 保存最佳模型
        if avg_test < best_loss:
            best_loss = avg_test
            torch.save(model.state_dict(), 'best_model.pt')

        if (epoch + 1) % 20 == 0:
            print(f'Epoch [{epoch + 1}/100], Train Loss: {avg_train:.4f}, Test Loss: {avg_test:.4f}')

    return train_losses, test_losses


# 6. 可视化函数
def plot_results(train_losses, test_losses):
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='训练损失', linewidth=2)
    plt.plot(test_losses, label='测试损失', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('训练和测试损失')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.plot(test_losses, color='red', linewidth=2, label='测试损失')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('测试损失变化')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('training_plot.png', dpi=300, bbox_inches='tight')
    plt.show()


# 7. 主函数
def main():
    # 加载数据
    X, y, scaler_X, scaler_y = dataloader()
    if X is None:
        return

    # 划分数据集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 创建数据集
    train_dataset = CountriesDataset(X_train, y_train)
    test_dataset = CountriesDataset(X_test, y_test)

    # 配置DataLoader
    batch_size = 32
    num_workers = 2

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    print(f"\n数据集信息:")
    print(f"训练样本: {len(train_dataset)}")
    print(f"测试样本: {len(test_dataset)}")
    print(f"Batch size: {batch_size}")
    print(f"Workers: {num_workers}")

    # 设备设置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 创建模型
    input_size = X_train.shape[1]
    model = FiveLayerNetwork(input_size).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型总参数: {total_params}")

    # 训练模型
    train_losses, test_losses = train_model(model, train_loader, test_loader, device)

    # 可视化结果
    plot_results(train_losses, test_losses)

    # 加载最佳模型进行最终评估
    model.load_state_dict(torch.load('best_model.pt'))
    model.eval()

    # 最终评估
    criterion = nn.MSELoss()
    final_test_loss = 0
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            final_test_loss += loss.item()

    print(f"\n训练完成!")
    print(f"最终测试损失: {final_test_loss / len(test_loader):.6f}")
    print("最佳模型已保存为: best_model.pt")
    print("训练图像已保存为: training_plot.png")


if __name__ == "__main__":
    main()
