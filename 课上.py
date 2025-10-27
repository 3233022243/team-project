import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import matplotlib

# 解决中文显示问题（关键修正）
# 修改后（仅保留系统默认黑体，避免找不到字体）
matplotlib.rcParams["font.family"] = "SimHei"  # 仅保留Windows自带的SimHei
matplotlib.rcParams["axes.unicode_minus"] = False
# 设置随机种子确保可复现性
torch.manual_seed(42)
np.random.seed(42)


# 1. 数据加载与清洗
class StrokeDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32).unsqueeze(1)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


# 加载数据
df = pd.read_csv("healthcare-dataset-stroke-data.csv")

# 数据清洗前探索
print("===== 清洗前数据信息 =====")
print(f"样本数量：{df.shape[0]}, 特征数量：{df.shape[1] - 1}")
print("\n类别分布：")
print(df['stroke'].value_counts())
print("\n缺失值统计：")
print(df.isnull().sum())
print("\n数据类型：")
print(df.dtypes)

# 数据清洗步骤
cleaned_df = df.copy()

# 创建年龄分组特征
cleaned_df['age_group'] = pd.cut(
    cleaned_df['age'],
    bins=[0, 18, 30, 50, 70, 120],
    labels=['0-18', '19-30', '31-50', '51-70', '71+']
)

# 处理缺失值
cleaned_df['bmi'] = cleaned_df.groupby(['gender', 'age_group'])['bmi'].transform(
    lambda x: x.fillna(x.mean()) if not x.isna().all() else x.fillna(cleaned_df['bmi'].mean())
)
cleaned_df['bmi'].fillna(cleaned_df['bmi'].mean(), inplace=True)

# 处理异常值
cleaned_df = cleaned_df[(cleaned_df['age'] >= 0) & (cleaned_df['age'] <= 120)]
cleaned_df = cleaned_df[(cleaned_df['avg_glucose_level'] >= 0) & (cleaned_df['avg_glucose_level'] <= 300)]
cleaned_df = cleaned_df[(cleaned_df['bmi'] >= 10) & (cleaned_df['bmi'] <= 60)]

# 处理类别特征异常值
if 'Other' in cleaned_df['gender'].unique():
    cleaned_df['gender'] = cleaned_df['gender'].replace('Other', cleaned_df['gender'].mode()[0])

# 清洗后数据信息
print("\n===== 清洗后数据信息 =====")
print(f"样本数量：{cleaned_df.shape[0]}, 特征数量：{cleaned_df.shape[1] - 1}")
print("\n缺失值检查：")
print(cleaned_df.isnull().sum())
print("\n类别分布：")
print(cleaned_df['stroke'].value_counts())

# 分离特征和标签
X = cleaned_df.drop(['stroke', 'age_group'], axis=1)
y = cleaned_df['stroke'].values

# 区分数值和类别特征
numeric_features = ['age', 'hypertension', 'heart_disease', 'avg_glucose_level', 'bmi']
categorical_features = ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']

# 创建预处理管道
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(drop='first'), categorical_features)
    ])

# 预处理特征
X_processed = preprocessor.fit_transform(X)
print(f"\n预处理后特征维度：{X_processed.shape[1]}")

# 划分数据集
X_train, X_test, y_train, y_test = train_test_split(
    X_processed, y, test_size=0.2, random_state=42, stratify=y)

# 创建数据集实例
train_dataset = StrokeDataset(X_train, y_train)
test_dataset = StrokeDataset(X_test, y_test)

# 配置DataLoader
batch_size = 32
train_size = int(0.9 * len(train_dataset))
val_size = len(train_dataset) - train_size
train_subset, val_subset = random_split(
    train_dataset, [train_size, val_size],
    generator=torch.Generator().manual_seed(42)
)

train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=0)
val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

print(f"\n训练集样本数：{len(train_subset)}, 验证集样本数：{len(val_subset)}, 测试集样本数：{len(test_dataset)}")


# 构建模型
class StrokePredictionModel(nn.Module):
    def __init__(self, input_dim):
        super(StrokePredictionModel, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(32, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(16, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.layers(x)


# 初始化模型
input_dim = X_processed.shape[1]
model = StrokePredictionModel(input_dim)

# 配置训练参数
criterion = nn.BCELoss()
optimizer = optim.Adam(
    model.parameters(),
    lr=0.0005,
    weight_decay=1e-5
)
num_epochs = 50
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
print(f"\n使用设备：{device}")

clip_value = 1.0  # 梯度裁剪阈值


# 训练模型
def calculate_accuracy(y_pred, y_true):
    y_pred_tag = torch.round(y_pred)
    correct = (y_pred_tag == y_true).sum().float()
    return correct / len(y_true)


train_losses = []
train_accuracies = []
val_losses = []
val_accuracies = []
best_val_acc = 0.0
best_model_path = "best_stroke_model.pt"

for epoch in range(num_epochs):
    model.train()
    train_running_loss = 0.0
    train_running_acc = 0.0

    for features, labels in train_loader:
        features, labels = features.to(device), labels.to(device)
        outputs = model(features)
        loss = criterion(outputs, labels)
        acc = calculate_accuracy(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), clip_value)
        optimizer.step()

        train_running_loss += loss.item() * features.size(0)
        train_running_acc += acc.item() * features.size(0)

    train_epoch_loss = train_running_loss / len(train_subset)
    train_epoch_acc = train_running_acc / len(train_subset)
    train_losses.append(train_epoch_loss)
    train_accuracies.append(train_epoch_acc)

    model.eval()
    val_running_loss = 0.0
    val_running_acc = 0.0

    with torch.no_grad():
        for features, labels in val_loader:
            features, labels = features.to(device), labels.to(device)
            outputs = model(features)
            loss = criterion(outputs, labels)
            acc = calculate_accuracy(outputs, labels)

            val_running_loss += loss.item() * features.size(0)
            val_running_acc += acc.item() * features.size(0)

    val_epoch_loss = val_running_loss / len(val_subset)
    val_epoch_acc = val_running_acc / len(val_subset)
    val_losses.append(val_epoch_loss)
    val_accuracies.append(val_epoch_acc)

    if val_epoch_acc > best_val_acc:
        best_val_acc = val_epoch_acc
        torch.save(model.state_dict(), best_model_path)
        print(f"第{epoch + 1}轮保存最佳模型，验证准确率：{best_val_acc:.4f}")
    else:
        print(f"第{epoch + 1}轮 - 训练损失：{train_epoch_loss:.4f}, 训练准确率：{train_epoch_acc:.4f}, "
              f"验证损失：{val_epoch_loss:.4f}, 验证准确率：{val_epoch_acc:.4f}")

# 可视化训练过程（修正标签显示）
plt.figure(figsize=(12, 5))

# 损失曲线
plt.subplot(1, 2, 1)
plt.plot(range(1, num_epochs + 1), train_losses, label='训练损失')
plt.plot(range(1, num_epochs + 1), val_losses, label='验证损失')
plt.title('损失曲线')
plt.xlabel('轮次')
plt.ylabel('损失值')
plt.legend()

# 准确率曲线
plt.subplot(1, 2, 2)
plt.plot(range(1, num_epochs + 1), train_accuracies, label='训练准确率')
plt.plot(range(1, num_epochs + 1), val_accuracies, label='验证准确率')
plt.title('准确率曲线')
plt.xlabel('轮次')
plt.ylabel('准确率')
plt.legend()

plt.tight_layout()
plt.show()

# 测试集评估（解决模型加载警告）
best_model = StrokePredictionModel(input_dim)
# 关键修正：添加weights_only=True参数
best_model.load_state_dict(torch.load(best_model_path, weights_only=True))
best_model.to(device)
best_model.eval()

test_running_acc = 0.0
all_preds = []
all_labels = []

with torch.no_grad():
    for features, labels in test_loader:
        features, labels = features.to(device), labels.to(device)
        outputs = best_model(features)
        acc = calculate_accuracy(outputs, labels)
        test_running_acc += acc.item() * features.size(0)

        all_preds.extend(torch.round(outputs).cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

test_acc = test_running_acc / len(test_dataset)
print(f"\n最佳模型在测试集上的准确率：{test_acc:.4f}")

# 打印混淆矩阵
cm = confusion_matrix(all_labels, all_preds)
print("\n测试集混淆矩阵：")
print(cm)
print(f"真阳性：{cm[1, 1]}, 真阴性：{cm[0, 0]}")
print(f"假阳性：{cm[0, 1]}, 假阴性：{cm[1, 0]}")