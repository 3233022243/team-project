import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import warnings
import os  # 用于路径检查

warnings.filterwarnings("ignore")


def process_countries_data(csv_path):
    """处理countries数据集：加载、清洗、缺失值填充、特征标签分离"""
    # 检查文件是否存在，避免路径错误
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"文件不存在，请检查路径：{csv_path}")

    # 用传入的正确CSV路径加载数据
    df = pd.read_csv(csv_path)
    print(f"原始数据形状：{df.shape}（样本数：{df.shape[0]}, 列数：{df.shape[1]}）")

    # 筛选有用的数值列
    useful_columns = [
        "Region", "Population (millions)", "HDI", "GDP per Capita",
        "Cropland Footprint", "Grazing Footprint", "Forest Footprint",
        "Carbon Footprint", "Fish Footprint", "Total Ecological Footprint",
        "Cropland", "Grazing Land", "Forest Land", "Fishing Water",
        "Urban Land", "Total Biocapacity", "Biocapacity Deficit or Reserve"
    ]
    # 确保数据集包含所需列，避免KeyError
    missing_cols = [col for col in useful_columns if col not in df.columns]
    if missing_cols:
        raise ValueError(f"数据集缺少必要列：{missing_cols}")
    df = df[useful_columns].copy()

    # 清洗GDP字段（处理$和逗号，转为纯浮点数）
    df["GDP per Capita"] = df["GDP per Capita"].str.replace("$", "").str.replace(",", "").astype(float)

    # 缺失值处理（按Region分组填充，更贴合数据逻辑）
    print("\n缺失值统计（处理前）：")
    print(df.isnull().sum()[df.isnull().sum() > 0])
    for col in df.columns:
        if col != "Region" and df[col].dtype in [np.float64, np.float32]:
            df[col] = df.groupby("Region")[col].transform(lambda x: x.fillna(x.mean()))
            df[col] = df[col].fillna(df[col].mean())  # 兜底：用全局均值填充剩余缺失值
    print("\n缺失值统计（处理后）：")
    print(df.isnull().sum()[df.isnull().sum() > 0])

    # 特征与标签分离（标签：生物承载力盈余=1，赤字=0）
    X = df.drop(["Region", "Biocapacity Deficit or Reserve"], axis=1)
    y = (df["Biocapacity Deficit or Reserve"] > 0).astype(int).values.reshape(-1, 1)

    # 特征标准化（消除量纲影响，加速模型收敛）
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    input_dim = X_scaled.shape[1]
    print(f"\n数据处理完成：特征维度{X_scaled.shape}，标签维度{y.shape}，输入层维度{input_dim}")
    return X_scaled, y, input_dim


# ---------------------- 关键：替换为你的CSV文件路径 ----------------------
csv_path = r"D:\python\神经网络\countries.csv"
X_scaled, y, input_dim = process_countries_data(csv_path)


class CountriesDataset(Dataset):
    """国家生态足迹数据集类（适配PyTorch DataLoader）"""

    def __init__(self, features, labels):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32)  # BCELoss需float类型标签
        self.length = features.shape[0]

    def __getitem__(self, index):
        return self.features[index], self.labels[index]

    def __len__(self):
        return self.length


def get_dataloaders(features, labels, batch_size=8, num_workers=0):
    """生成训练/测试DataLoader（Windows建议num_workers=0，避免多线程报错）"""
    full_dataset = CountriesDataset(features, labels)
    # 划分训练集（80%）和测试集（20%），固定随机种子确保结果可复现
    train_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )

    # 训练集：打乱数据提升泛化能力
    train_loader = DataLoader(
        dataset=train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, drop_last=False
    )
    # 测试集：不打乱，便于稳定评估
    test_loader = DataLoader(
        dataset=test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers
    )

    print(f"\nDataLoader初始化完成：")
    print(
        f"训练集：{len(train_dataset)}样本，{len(train_loader)}批次；测试集：{len(test_dataset)}样本，{len(test_loader)}批次")
    return train_loader, test_loader


# ---------------------- 初始化DataLoader ----------------------
train_loader, test_loader = get_dataloaders(
    features=X_scaled, labels=y, batch_size=8, num_workers=0
)


class FiveLayerNN(nn.Module):
    """5层神经网络（含权重初始化，防止梯度爆炸）"""

    def __init__(self, input_dim):
        super(FiveLayerNN, self).__init__()
        # 网络结构：输入层→隐藏层1(7)→隐藏层2(6)→隐藏层3(5)→输出层(1)
        self.linear1 = nn.Linear(input_dim, 7)
        self.linear2 = nn.Linear(7, 6)
        self.linear3 = nn.Linear(6, 5)
        self.linear4 = nn.Linear(5, 1)

        # 激活函数：ReLU缓解梯度消失，Sigmoid映射输出到0-1（二分类概率）
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

        # 权重初始化（Xavier初始化，适配ReLU，避免初始权重过大）
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)  # 均匀初始化权重
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.1)  # 偏置初始化为0.1，避免神经元死亡

    def forward(self, x):
        # 前向传播路径
        x = self.relu(self.linear1(x))
        x = self.relu(self.linear2(x))
        x = self.relu(self.linear3(x))
        x = self.sigmoid(self.linear4(x))
        return x


# ---------------------- 实例化5层模型 ----------------------
model = FiveLayerNN(input_dim=input_dim)
print("\n5层神经网络结构：")
print(model)


def train_model(model, train_loader, test_loader, epochs=50, lr=0.001,
                save_path=r"C:\Users\dell\Downloads\best_countries_model.pt"):
    """模型训练（含梯度裁剪，防止梯度爆炸；L2正则化，抑制过拟合）"""
    # 损失函数：二分类交叉熵（适配Sigmoid输出）
    criterion = nn.BCELoss()
    # 优化器：Adam（收敛快）+ L2正则化（weight_decay=1e-5）
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

    # 训练记录（用于后续可视化）
    train_loss_log = []
    test_loss_log = []
    test_acc_log = []
    best_acc = 0.0  # 记录最优测试准确率

    for epoch in range(epochs):
        # ---------------------- 训练阶段 ----------------------
        model.train()  # 开启训练模式（启用Dropout等，此处无但保留规范）
        train_total_loss = 0.0
        for batch_idx, (inputs, labels) in enumerate(train_loader):
            # 前向传播：计算预测值
            outputs = model(inputs)
            # 计算损失
            loss = criterion(outputs, labels)

            # 反向传播+参数更新
            optimizer.zero_grad()  # 清空上一轮梯度
            loss.backward()  # 反向传播计算梯度
            # 梯度裁剪：最大范数1.0，超过则缩放，防止梯度爆炸
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()  # 更新模型参数

            # 累加训练损失（按样本数加权）
            train_total_loss += loss.item() * inputs.size(0)

        # 计算训练集每轮平均损失
        train_avg_loss = train_total_loss / len(train_loader.dataset)
        train_loss_log.append(train_avg_loss)

        # ---------------------- 测试阶段（无梯度计算） ----------------------
        model.eval()  # 开启评估模式（关闭Dropout等）
        test_total_loss = 0.0
        correct = 0  # 正确预测数
        total = 0  # 总测试样本数
        with torch.no_grad():  # 禁用梯度计算，节省内存+加速
            for inputs, labels in test_loader:
                outputs = model(inputs)
                # 计算测试损失
                loss = criterion(outputs, labels)
                test_total_loss += loss.item() * inputs.size(0)
                # 计算准确率（输出>0.5视为预测1，否则0）
                predicted = (outputs > 0.5).float()
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        # 计算测试集每轮平均损失和准确率
        test_avg_loss = test_total_loss / len(test_loader.dataset)
        test_acc = correct / total
        test_loss_log.append(test_avg_loss)
        test_acc_log.append(test_acc)

        # 打印每轮训练日志
        print(f"Epoch [{epoch + 1}/{epochs}] | "
              f"Train Loss: {train_avg_loss:.4f} | "
              f"Test Loss: {test_avg_loss:.4f} | "
              f"Test Acc: {test_acc:.4f}")

        # 保存测试准确率最高的模型（仅保存最优模型，避免过拟合模型）
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), save_path)
            print(f" 最优模型更新！准确率：{best_acc:.4f}，已保存至 {save_path}")

    return train_loss_log, test_loss_log, test_acc_log, best_acc


# ---------------------- 启动模型训练 ----------------------
train_loss, test_loss, test_acc, best_accuracy = train_model(
    model=model, train_loader=train_loader, test_loader=test_loader,
    epochs=50, lr=0.001
)
print(f"\n训练结束！最优测试准确率：{best_accuracy:.4f}，模型已保存至 Downloads 文件夹")


def plot_training_curve(train_loss, test_loss, test_acc, save_path=r"C:\Users\dell\Downloads\training_curve.png"):
    """绘制训练曲线（损失+准确率），保存至Downloads文件夹"""
    epochs = len(train_loss)
    # 创建1行2列子图，便于对比
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # 子图1：训练/测试损失曲线
    ax1.plot(range(1, epochs + 1), train_loss, label="Train Loss", color="#1f77b4", linewidth=2)
    ax1.plot(range(1, epochs + 1), test_loss, label="Test Loss", color="#ff7f0e", linewidth=2)
    ax1.set_xlabel("Epoch", fontsize=12)
    ax1.set_ylabel("BCELoss", fontsize=12)
    ax1.set_title("Training & Test Loss Curve", fontsize=14, fontweight="bold")
    ax1.legend(fontsize=10)
    ax1.grid(alpha=0.3)

    # 子图2：测试集准确率曲线
    ax2.plot(range(1, epochs + 1), test_acc, label="Test Accuracy", color="#2ca02c", linewidth=2)
    ax2.set_xlabel("Epoch", fontsize=12)
    ax2.set_ylabel("Accuracy", fontsize=12)
    ax2.set_ylim(0, 1.05)  # 固定准确率范围0-1.05，便于观察
    ax2.set_title("Test Accuracy Curve", fontsize=14, fontweight="bold")
    ax2.legend(fontsize=10)
    ax2.grid(alpha=0.3)

    # 保存图像（高分辨率+避免文字截断）
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()
    print(f"训练曲线已保存至：{save_path}")


# ---------------------- 绘制并保存训练曲线 ----------------------
plot_training_curve(train_loss=train_loss, test_loss=test_loss, test_acc=test_acc)