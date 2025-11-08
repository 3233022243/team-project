import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

# ... 其余代码（配置参数、数据加载、模型定义等）
# -------------------------- 1. 配置参数 --------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 优先使用GPU
batch_size = 64
learning_rate = 0.001
epochs = 10  # 训练轮次

# -------------------------- 2. 数据加载与预处理 --------------------------
# 数据预处理：归一化到[0,1]，转为Tensor
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))  # MNIST数据集的均值和标准差
])

# 加载MNIST数据集（自动下载到./data目录）
train_dataset = datasets.MNIST(
    root='./data', train=True, download=True, transform=transform
)
test_dataset = datasets.MNIST(
    root='./data', train=False, download=True, transform=transform
)

# 创建数据加载器
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


# -------------------------- 3. 定义模型（FC和CNN） --------------------------
# 3.1 全连接神经网络（FC）
class FCNet(nn.Module):
    def __init__(self):
        super(FCNet, self).__init__()
        # 输入：28*28=784个像素，隐藏层128，输出10个类别（0-9）
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)
        self.relu = nn.ReLU()  # 激活函数

    def forward(self, x):
        x = x.view(-1, 784)  # 展平：(batch_size, 1, 28, 28) → (batch_size, 784)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)  # 输出层不激活（交给CrossEntropyLoss）
        return x


# 3.2 卷积神经网络（CNN）
class CNNNet(nn.Module):
    def __init__(self):
        super(CNNNet, self).__init__()
        # 卷积层1：输入1通道（灰度图），输出16通道，卷积核3x3，步长1， padding=1
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        # 卷积层2：输入16通道，输出32通道，卷积核3x3
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        # 池化层：2x2最大池化
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        # 全连接层1：32通道 * 7x7特征图（28→14→7）= 1568
        self.fc1 = nn.Linear(32 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        # 卷积1 → 激活 → 池化：(batch,1,28,28)→(batch,16,28,28)→(batch,16,14,14)
        x = self.pool(self.relu(self.conv1(x)))
        # 卷积2 → 激活 → 池化：(batch,16,14,14)→(batch,32,14,14)→(batch,32,7,7)
        x = self.pool(self.relu(self.conv2(x)))
        # 展平：(batch, 32*7*7)
        x = x.view(-1, 32 * 7 * 7)
        # 全连接层
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# -------------------------- 4. 训练与测试函数 --------------------------
def train_model(model, train_loader, criterion, optimizer, epoch):
    """训练单个轮次"""
    model.train()  # 训练模式（启用Dropout等）
    running_loss = 0.0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)  # 数据移到GPU/CPU

        optimizer.zero_grad()  # 清空梯度
        output = model(data)  # 前向传播
        loss = criterion(output, target)  # 计算损失
        loss.backward()  # 反向传播
        optimizer.step()  # 更新参数

        running_loss += loss.item()
    # 返回本轮平均损失
    return running_loss / len(train_loader)


def test_model(model, test_loader, criterion):
    """测试模型，返回准确率和损失"""
    model.eval()  # 测试模式（禁用Dropout等）
    test_loss = 0.0
    correct = 0  # 正确预测的样本数

    with torch.no_grad():  # 禁用梯度计算（节省内存）
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()  # 累积损失
            pred = output.argmax(dim=1, keepdim=True)  # 预测类别（概率最大的）
            correct += pred.eq(target.view_as(pred)).sum().item()  # 累积正确数

    test_loss /= len(test_loader)
    test_acc = correct / len(test_loader.dataset)  # 准确率
    return test_acc, test_loss


# -------------------------- 5. 初始化模型、损失函数、优化器 --------------------------
# 全连接模型
fc_model = FCNet().to(device)
fc_criterion = nn.CrossEntropyLoss()  # 交叉熵损失（适用于分类）
fc_optimizer = optim.Adam(fc_model.parameters(), lr=learning_rate)  # Adam优化器

# 卷积模型
cnn_model = CNNNet().to(device)
cnn_criterion = nn.CrossEntropyLoss()
cnn_optimizer = optim.Adam(cnn_model.parameters(), lr=learning_rate)

# -------------------------- 6. 训练与记录数据 --------------------------
# 记录每轮的准确率
fc_train_accs = []
fc_test_accs = []
cnn_train_accs = []
cnn_test_accs = []

print("开始训练（共{}轮）...".format(epochs))
for epoch in range(1, epochs + 1):
    print(f"\n=== 第{epoch}轮 ===")

    # 训练FC模型
    fc_train_loss = train_model(fc_model, train_loader, fc_criterion, fc_optimizer, epoch)
    fc_train_acc, _ = test_model(fc_model, train_loader, fc_criterion)  # 训练集准确率
    fc_test_acc, fc_test_loss = test_model(fc_model, test_loader, fc_criterion)  # 测试集准确率
    fc_train_accs.append(fc_train_acc)
    fc_test_accs.append(fc_test_acc)
    print(f"FC模型 - 训练损失：{fc_train_loss:.4f} | 训练准确率：{fc_train_acc:.4f} | 测试准确率：{fc_test_acc:.4f}")

    # 训练CNN模型
    cnn_train_loss = train_model(cnn_model, train_loader, cnn_criterion, cnn_optimizer, epoch)
    cnn_train_acc, _ = test_model(cnn_model, train_loader, cnn_criterion)
    cnn_test_acc, cnn_test_loss = test_model(cnn_model, test_loader, cnn_criterion)
    cnn_train_accs.append(cnn_train_acc)
    cnn_test_accs.append(cnn_test_acc)
    print(f"CNN模型 - 训练损失：{cnn_train_loss:.4f} | 训练准确率：{cnn_train_acc:.4f} | 测试准确率：{cnn_test_acc:.4f}")

# -------------------------- 7. 绘制准确率曲线 --------------------------
plt.figure(figsize=(10, 6))
# 绘制训练集准确率
plt.plot(range(1, epochs + 1), fc_train_accs, label='FC - 训练准确率', marker='o', linestyle='-', color='orange')
plt.plot(range(1, epochs + 1), cnn_train_accs, label='CNN - 训练准确率', marker='s', linestyle='-', color='blue')
# 绘制测试集准确率
plt.plot(range(1, epochs + 1), fc_test_accs, label='FC - 测试准确率', marker='o', linestyle='--', color='orange')
plt.plot(range(1, epochs + 1), cnn_test_accs, label='CNN - 测试准确率', marker='s', linestyle='--', color='blue')

# 图表美化
plt.xlabel('训练轮次（Epoch）', fontsize=12)
plt.ylabel('准确率（Accuracy）', fontsize=12)
plt.title('MNIST手写数字识别：CNN vs FC 准确率曲线对比', fontsize=14)
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)
plt.xticks(range(1, epochs + 1))  # x轴显示每一轮
plt.ylim(0.8, 1.0)  # y轴范围（突出差异）

# 保存图片（保存到当前目录）
plt.savefig('mnist_cnn_fc_accuracy_curve.png', dpi=300, bbox_inches='tight')
print("\n准确率曲线已保存为：mnist_cnn_fc_accuracy_curve.png")

# 显示图片（PyCharm中需开启交互式绘图）
plt.show()
