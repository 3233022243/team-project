import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import os

# 1. 数据预处理与加载
os.environ['TORCH_HOME'] = './torch_cache'  # 模型缓存目录
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.1307,), std=(0.3081,))  # MNIST标准化参数
])

# 加载数据集并划分训练集、验证集、测试集
full_train_dataset = datasets.MNIST(
    root='./mnist_data',
    train=True,
    transform=transform,
    download=True  # 首次运行自动下载，已下载可设为False
)
train_dataset = Subset(full_train_dataset, range(55000))  # 训练集55000条
val_dataset = Subset(full_train_dataset, range(55000, 60000))  # 验证集5000条
test_dataset = datasets.MNIST(
    root='./mnist_data',
    train=False,
    transform=transform,
    download=True
)

# 数据加载器
train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=256,
    shuffle=True,
    num_workers=0
)
val_loader = DataLoader(
    dataset=val_dataset,
    batch_size=256,
    shuffle=False,
    num_workers=0
)
test_loader = DataLoader(
    dataset=test_dataset,
    batch_size=256,
    shuffle=False,
    num_workers=0
)


# 2. 全连接网络（FC）定义
class FCNet(nn.Module):
    def __init__(self):
        super(FCNet, self).__init__()
        self.fc1 = nn.Linear(28 * 28, out_features=512)
        self.fc2 = nn.Linear(in_features=512, out_features=256)
        self.fc3 = nn.Linear(in_features=256, out_features=10)

    def forward(self, x):
        x = x.view(-1, 28 * 28)  # 展平为784维向量
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# 3. 卷积神经网络（CNN）定义
class CNNNet(nn.Module):
    def __init__(self):
        super(CNNNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)  # 修正拼写错误
        self.fc1 = nn.Linear(64 * 7 * 7, out_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # 卷积+激活+池化
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)  # 展平特征图
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# 4. 训练与测试函数
def train(model, train_loader, val_loader, optimizer, criterion, epoch, device):
    model.train()
    total_loss = 0.0
    for batch_idx, (data, target) in enumerate(train_loader):  # 修正变量名错误
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()  # 梯度清零
        output = model(data)
        loss = criterion(output, target)
        loss.backward()  # 修正大小写错误
        optimizer.step()  # 参数更新
        total_loss += loss.item()

        # 每150批次打印训练损失
        if batch_idx % 150 == 149:
            avg_loss = total_loss / 150
            print(f'[{epoch + 1}] 批次 {batch_idx + 1}: 训练损失= {avg_loss:.4f}')
            total_loss = 0.0

    # 训练完一轮后验证
    val_acc = test(model, val_loader, is_val=True, device=device)
    return val_acc


def test(model, data_loader, is_val=False, device=None):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():  # 关闭梯度计算
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

    accuracy = 100 * correct / total
    if is_val:
        print(f'验证准确率: {accuracy:.2f}%')
    else:
        print(f'测试准确率: {accuracy:.2f}% ({correct}/{total})\n')  # 修正变量名错误
    return accuracy


# 5. 主程序（模型训练与测试）
if __name__ == "__main__":
    # 设置设备（GPU优先）
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 初始化模型、损失函数、优化器
    fc_model = FCNet().to(device)
    cnn_model = CNNNet().to(device)
    criterion = nn.CrossEntropyLoss()  # 交叉熵损失（适用于分类任务）

    # 优化器配置
    fc_optimizer = torch.optim.Adam(fc_model.parameters(), lr=0.0008, weight_decay=1e-5)  # 修正拼写错误
    cnn_optimizer = torch.optim.Adam(cnn_model.parameters(), lr=0.0008, weight_decay=1e-5)

    # 训练参数
    epochs = 8  # 训练8轮

    # 训练FC模型
    print("=" * 50)
    print("开始训练全连接网络（FC）...")
    for epoch in range(epochs):
        train(fc_model, train_loader, val_loader, fc_optimizer, criterion, epoch, device)
    print("FC模型训练完成，开始测试...")
    fc_test_acc = test(fc_model, test_loader, is_val=False, device=device)

    # 训练CNN模型
    print("=" * 50)
    print("开始训练卷积神经网络（CNN）...")
    for epoch in range(epochs):
        train(cnn_model, train_loader, val_loader, cnn_optimizer, criterion, epoch, device)
    print("CNN模型训练完成，开始测试...")
    cnn_test_acc = test(cnn_model, test_loader, is_val=False, device=device)

    # 输出最终对比结果
    print("=" * 50)
    print(f"最终测试准确率对比：")
    print(f"全连接网络（FC）: {fc_test_acc:.2f}%")
    print(f"卷积神经网络（CNN）: {cnn_test_acc:.2f}%")