import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
import os
import re

torch.manual_seed(42)
np.random.seed(42)


# 1. 自定义Dataset（保持不变）
class CountryDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.tensor(features.astype(np.float32))
        self.labels = torch.tensor(labels.astype(np.float32))

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


# 2. 数据预处理（重点：目标变量对数变换，压缩尺度）
def load_and_preprocess_data(file_path):
    df = pd.read_csv(file_path)
    print(f"原始数据集形状: {df.shape}")

    target_col = 'GDP per Capita'

    # 清洗货币符号
    def clean_currency(value):
        if pd.isna(value):
            return np.nan
        cleaned = re.sub(r'[^\d.]', '', str(value))
        try:
            return float(cleaned)
        except:
            return np.nan

    df[target_col] = df[target_col].apply(clean_currency)
    print(f"目标变量货币符号清洗完成")

    # 处理全缺失（用生态足迹替代）
    if df[target_col].isnull().sum() == len(df):
        print(f"GDP列全缺失，使用Total Ecological Footprint作为模拟目标")
        df[target_col] = df['Total Ecological Footprint'].copy()

    # 关键：目标变量对数变换（压缩到小范围，避免MSE爆炸）
    df[target_col] = np.log1p(df[target_col])  # log(1 + x)，避免x=0时log(0)错误
    df[target_col].fillna(df[target_col].median(), inplace=True)

    X = df.drop([target_col, 'Country'], axis=1)
    y = df[target_col].values.astype(np.float32)

    # 特征预处理管道
    numeric_features = X.select_dtypes(include=['float64', 'int64']).columns.tolist()
    categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()

    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])

    X_processed = preprocessor.fit_transform(X)

    # 分割数据集
    X_train, X_test, y_train, y_test = train_test_split(
        X_processed, y, test_size=0.2, random_state=42
    )

    print(f"预处理完成:")
    print(f"  训练集：特征{X_train.shape} | 标签{y_train.shape}")
    print(f"  测试集：特征{X_test.shape} | 标签{y_test.shape}")
    return X_train, X_test, y_train, y_test


# 3. 5层神经网络（修改：用LeakyReLU替代ReLU，避免神经元死亡）
class CountryNN(nn.Module):
    def __init__(self, input_size):
        super(CountryNN, self).__init__()
        self.layers = nn.Sequential(
            # 输入→隐藏1（7神经元）
            nn.Linear(input_size, 7),
            nn.BatchNorm1d(7),
            nn.LeakyReLU(0.1),  # 关键：LeakyReLU允许负输入有小梯度，避免死亡
            nn.Dropout(0.2),

            # 隐藏1→隐藏2（6神经元）
            nn.Linear(7, 6),
            nn.BatchNorm1d(6),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.2),

            # 隐藏2→隐藏3（5神经元）
            nn.Linear(6, 5),
            nn.BatchNorm1d(5),
            nn.LeakyReLU(0.1),

            # 隐藏3→输出
            nn.Linear(5, 1)
        )
        # Kaiming初始化（适配LeakyReLU）
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, a=0.1, mode='fan_in', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return self.layers(x)


# 4. 模型训练（修改：学习率调大+余弦退火，动态调整学习率）
def train_model(model, train_loader, test_loader, criterion, optimizer, device, num_epochs=150, patience=15):
    train_losses = []
    test_losses = []
    best_test_loss = float('inf')
    early_stop_counter = 0

    # 初始保存模型
    torch.save(model.state_dict(), 'best_model.pt')

    # 余弦退火学习率调度器（动态调整学习率，避免停滞）
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0

        for features, labels in train_loader:
            features, labels = features.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(features).squeeze()
            loss = criterion(outputs, labels)

            # 梯度裁剪（防爆炸）
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            loss.backward()
            optimizer.step()

            train_loss += loss.item() * features.size(0)

        avg_train_loss = train_loss / len(train_loader.dataset)
        train_losses.append(avg_train_loss)

        # 测试集评估
        model.eval()
        test_loss = 0.0
        with torch.no_grad():
            for features, labels in test_loader:
                features, labels = features.to(device), labels.to(device)
                outputs = model(features).squeeze()
                test_loss += criterion(outputs, labels).item() * features.size(0)

        avg_test_loss = test_loss / len(test_loader.dataset)
        test_losses.append(avg_test_loss)

        # 打印进度
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch + 1}/{num_epochs}] | 训练损失: {avg_train_loss:.4f} | 测试损失: {avg_test_loss:.4f}")

        # 保存最优模型
        if avg_test_loss < best_test_loss:
            best_test_loss = avg_test_loss
            torch.save(model.state_dict(), 'best_model.pt')
            early_stop_counter = 0
        else:
            early_stop_counter += 1
            if early_stop_counter >= patience:
                print(f"早停触发：第{epoch + 1}轮，测试损失连续{patience}轮未改善")
                break

        # 学习率调度
        scheduler.step()

    # 加载最优模型
    try:
        model.load_state_dict(torch.load('best_model.pt', weights_only=True))
    except TypeError:
        model.load_state_dict(torch.load('best_model.pt'))

    return model, train_losses, test_losses


# 5. 可视化（保持不变）
def plot_training_history(train_losses, test_losses):
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(train_losses) + 1), train_losses,
             color='#1f77b4', marker='o', markersize=4, linewidth=2, label='训练损失')
    plt.plot(range(1, len(test_losses) + 1), test_losses,
             color='#ff7f0e', marker='s', markersize=4, linewidth=2, label='测试损失')

    plt.title('全连接神经网络训练过程损失变化', fontsize=14, pad=15)
    plt.xlabel('训练轮次（Epoch）', fontsize=12)
    plt.ylabel('损失值（MSE）', fontsize=12)
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.legend(fontsize=11, loc='upper right')
    plt.tight_layout()
    plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("训练可视化图表已保存：training_history.png")


# 6. 主函数
def main():
    file_path = r"C:\Users\dell\Downloads\countries-1760073842817.csv"

    if not os.path.exists(file_path):
        print(f"数据集不存在：{file_path}")
        return

    # 数据预处理（含目标变量对数变换）
    X_train, X_test, y_train, y_test = load_and_preprocess_data(file_path)

    # DataLoader配置（保持不变）
    train_dataset = CountryDataset(X_train, y_train)
    test_dataset = CountryDataset(X_test, y_test)

    batch_size = 16
    num_workers = 0
    print(f"\nDataLoader配置：batch_size={batch_size}, num_workers={num_workers}")

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    # 模型初始化
    input_size = X_train.shape[1]
    device = torch.device('cpu')
    print(f"\n使用计算设备：{device}")
    model = CountryNN(input_size).to(device)

    # 损失函数与优化器（学习率调大到0.01，配合余弦退火）
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-5)

    # 模型训练（轮次增加到150）
    print("\n开始训练...")
    trained_model, train_losses, test_losses = train_model(
        model, train_loader, test_loader, criterion, optimizer, device,
        num_epochs=150, patience=15
    )

    # 可视化与最终评估
    plot_training_history(train_losses, test_losses)

    trained_model.eval()
    with torch.no_grad():
        test_features = torch.tensor(X_test, dtype=torch.float32).to(device)
        test_labels = torch.tensor(y_test, dtype=torch.float32).to(device)
        outputs = trained_model(test_features).squeeze()
        final_loss = criterion(outputs, test_labels)

    print(f"\n最终测试损失：{final_loss.item():.4f}")
    print("最优模型已保存：best_model.pt")


if __name__ == "__main__":
    main()