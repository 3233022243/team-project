import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import warnings

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False    # 用来正常显示负号
warnings.filterwarnings('ignore')

# 设置随机种子以保证结果可重现
torch.manual_seed(42)
np.random.seed(42)


# 1. 数据预处理类
class StrokeDataset(Dataset):
    def __init__(self, features, targets):
        self.features = torch.FloatTensor(features)
        self.targets = torch.LongTensor(targets)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]


# 2. 神经网络模型定义
class StrokeNet(nn.Module):
    def __init__(self, input_size):
        super(StrokeNet, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Dropout(0.1),

            nn.Linear(16, 2)  # 二分类输出
        )

    def forward(self, x):
        return self.network(x)


# 3. 数据加载和预处理函数
def load_and_preprocess_data(file_path='healthcare-dataset-stroke-data.csv'):
    # 读取数据
    df = pd.read_csv(file_path)
    print(f"数据集形状: {df.shape}")

    # 显示目标变量分布
    print(f"\n目标变量分布:")
    print(df['stroke'].value_counts())
    print(f"中风比例: {df['stroke'].mean():.4f}")

    # 数据预处理
    df_processed = df.copy()

    # 处理bmi列的缺失值，使用中位数填充
    if 'bmi' in df_processed.columns:
        bmi_median = df_processed['bmi'].median()
        df_processed['bmi'].fillna(bmi_median, inplace=True)

    # 特征工程
    categorical_columns = ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']
    numerical_columns = ['age', 'hypertension', 'heart_disease', 'avg_glucose_level', 'bmi']

    # 对分类变量进行编码
    label_encoders = {}
    for col in categorical_columns:
        if col in df_processed.columns:
            le = LabelEncoder()
            df_processed[col] = le.fit_transform(df_processed[col].astype(str))
            label_encoders[col] = le

    # 选择最终特征
    feature_columns = numerical_columns + categorical_columns
    feature_columns = [col for col in feature_columns if col in df_processed.columns]

    print(f"\n使用的特征列 ({len(feature_columns)}个): {feature_columns}")

    # 提取特征和目标
    X = df_processed[feature_columns].values
    y = df_processed['stroke'].values

    # 数据标准化
    scaler_X = StandardScaler()
    X_scaled = scaler_X.fit_transform(X)

    return X_scaled, y, scaler_X, label_encoders, feature_columns


# 4. 训练函数
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=100):
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    best_val_accuracy = 0.0
    best_model_state = None

    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for batch_features, batch_targets in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_features)
            loss = criterion(outputs, batch_targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += batch_targets.size(0)
            train_correct += (predicted == batch_targets).sum().item()

        # 验证阶段
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for batch_features, batch_targets in val_loader:
                outputs = model(batch_features)
                loss = criterion(outputs, batch_targets)
                val_loss += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                val_total += batch_targets.size(0)
                val_correct += (predicted == batch_targets).sum().item()

        # 计算指标
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        train_accuracy = 100 * train_correct / train_total
        val_accuracy = 100 * val_correct / val_total

        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        train_accuracies.append(train_accuracy)
        val_accuracies.append(val_accuracy)

        # 保存最佳模型
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_model_state = model.state_dict().copy()

        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}]')
            print(f'  训练损失: {avg_train_loss:.4f}, 训练准确率: {train_accuracy:.2f}%')
            print(f'  验证损失: {avg_val_loss:.4f}, 验证准确率: {val_accuracy:.2f}%')
            print('-' * 50)

    return train_losses, val_losses, train_accuracies, val_accuracies, best_model_state


# 5. 可视化函数
def plot_training_curves(train_losses, val_losses, train_accuracies, val_accuracies):
    # 创建子图
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # 损失曲线
    ax1.plot(train_losses, label='训练损失', linewidth=2, color='blue')
    ax1.plot(val_losses, label='验证损失', linewidth=2, color='red')
    ax1.set_xlabel('训练轮次')
    ax1.set_ylabel('损失值')
    ax1.set_title('训练和验证损失曲线')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 准确率曲线
    ax2.plot(train_accuracies, label='训练准确率', linewidth=2, color='blue')
    ax2.plot(val_accuracies, label='验证准确率', linewidth=2, color='red')
    ax2.set_xlabel('训练轮次')
    ax2.set_ylabel('准确率 (%)')
    ax2.set_title('训练和验证准确率曲线')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('training_curves.png', dpi=300, bbox_inches='tight')
    plt.show()


# 6. 主函数
def main():
    # 参数设置
    BATCH_SIZE = 32
    NUM_EPOCHS = 100
    LEARNING_RATE = 0.001

    print("开始处理中风预测数据集...")

    # 加载和预处理数据
    X, y, scaler_X, label_encoders, feature_columns = load_and_preprocess_data('healthcare-dataset-stroke-data.csv')

    print(f"\n处理后数据形状 - 特征: {X.shape}, 目标: {y.shape}")
    print(f"输入特征维度: {X.shape[1]}")
    print(f"类别分布 - 无中风: {np.sum(y == 0)}, 有中风: {np.sum(y == 1)}")

    # 划分训练集和验证集
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # 创建数据集
    train_dataset = StrokeDataset(X_train, y_train)
    val_dataset = StrokeDataset(X_val, y_val)

    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    print(f"\n数据加载器配置:")
    print(f"  批次大小: {BATCH_SIZE}")
    print(f"  训练批次数量: {len(train_loader)}")
    print(f"  验证批次数量: {len(val_loader)}")

    # 初始化模型
    input_size = X.shape[1]
    model = StrokeNet(input_size)

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)

    print(f"\n开始训练...")

    # 训练模型
    train_losses, val_losses, train_accuracies, val_accuracies, best_model_state = train_model(
        model, train_loader, val_loader, criterion, optimizer, NUM_EPOCHS
    )

    # 保存最佳模型
    torch.save({
        'model_state_dict': best_model_state,
        'scaler_X': scaler_X,
        'label_encoders': label_encoders,
        'feature_columns': feature_columns,
        'input_size': input_size
    }, 'best_stroke_model.pt')

    print(f"\n最佳模型已保存为 'best_stroke_model.pt'")

    # 绘制训练曲线
    plot_training_curves(train_losses, val_losses, train_accuracies, val_accuracies)

    # 最终评估
    model.load_state_dict(best_model_state)
    model.eval()

    # 在验证集上进行预测
    val_predictions = []
    val_targets = []

    with torch.no_grad():
        for batch_features, batch_targets in val_loader:
            outputs = model(batch_features)
            _, predicted = torch.max(outputs.data, 1)
            val_predictions.extend(predicted.numpy())
            val_targets.extend(batch_targets.numpy())

    # 计算准确率
    val_accuracy = accuracy_score(val_targets, val_predictions)

    print(f"\n模型性能评估:")
    print(f"验证集准确率: {val_accuracy:.4f}")
    print(f"\n分类报告:")
    print(classification_report(val_targets, val_predictions, target_names=['No Stroke', 'Stroke']))


if __name__ == "__main__":
    main()