import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# 数据预处理：归一化、转换为Tensor
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# 加载训练集和测试集
train_dataset = datasets.MNIST(root='./data', train=True, download=False, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=False, transform=transform)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1000, shuffle=False)


class ComplexCNN(nn.Module):
    def __init__(self):
        super(ComplexCNN, self).__init__()
        # 卷积层1：输入通道1，输出通道16，卷积核3x3
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        # 卷积层2：输入通道16，输出通道32，卷积核3x3
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        # 卷积层3：输入通道32，输出通道64，卷积核3x3
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)

        # 池化层：2x2最大池化
        self.pool = nn.MaxPool2d(2)

        # 线性层1：输入是池化后的特征数，输出128
        self.fc1 = nn.Linear(64 * 3 * 3, 128)  # 经过3次池化，28→14→7→3
        # 线性层2：输入128，输出64
        self.fc2 = nn.Linear(128, 64)
        # 线性层3：输入64，输出10（分类数）
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        # 卷积+ReLU+池化 模块1
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        # 卷积+ReLU+池化 模块2
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        # 卷积+ReLU+池化 模块3
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        # 展平特征图
        x = x.view(x.size(0), -1)
        # 线性层+ReLU
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        # 输出层
        x = self.fc3(x)
        return x

class FCNet(nn.Module):
    def __init__(self):
        super(FCNet, self).__init__()
        # 输入层：28×28=784个特征
        self.fc1 = nn.Linear(784, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 10)
    def forward(self, x):
        # 展平输入
        x = x.view(x.size(0), -1)
        # 线性层+ReLU
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        # 输出层
        x = self.fc3(x)
        return x
def train(model, train_loader, optimizer, criterion, epoch, device):
    model.train()
    train_loss = 0
    correct = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
    train_loss /= len(train_loader)
    train_acc = correct / len(train_loader.dataset)
    return train_loss, train_acc

def test(model, test_loader, criterion, device):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader)
    test_acc = correct / len(test_loader.dataset)
    return test_loss, test_acc

# 选择设备（GPU优先）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 初始化两个模型
cnn_model = ComplexCNN().to(device)
fc_model = FCNet().to(device)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
cnn_optimizer = optim.Adam(cnn_model.parameters(), lr=0.001)
fc_optimizer = optim.Adam(fc_model.parameters(), lr=0.001)

# 训练轮数
epochs = 15
# 记录训练过程的准确率和损失
cnn_train_losses = []
cnn_train_accs = []
cnn_test_losses = []
cnn_test_accs = []

fc_train_losses = []
fc_train_accs = []
fc_test_losses = []
fc_test_accs = []

# 训练并测试ComplexCNN
for epoch in range(1, epochs + 1):
    train_loss, train_acc = train(cnn_model, train_loader, cnn_optimizer, criterion, epoch, device)
    test_loss, test_acc = test(cnn_model, test_loader, criterion, device)
    cnn_train_losses.append(train_loss)
    cnn_train_accs.append(train_acc)
    cnn_test_losses.append(test_loss)
    cnn_test_accs.append(test_acc)
    print(f'CNN Epoch {epoch}: Train Loss={train_loss:.4f}, Train Acc={train_acc:.4f}, Test Loss={test_loss:.4f}, Test Acc={test_acc:.4f}')

# 训练并测试FCNet
for epoch in range(1, epochs + 1):
    train_loss, train_acc = train(fc_model, train_loader, fc_optimizer, criterion, epoch, device)
    test_loss, test_acc = test(fc_model, test_loader, criterion, device)
    fc_train_losses.append(train_loss)
    fc_train_accs.append(train_acc)
    fc_test_losses.append(test_loss)
    fc_test_accs.append(test_acc)
    print(f'FC Epoch {epoch}: Train Loss={train_loss:.4f}, Train Acc={train_acc:.4f}, Test Loss={test_loss:.4f}, Test Acc={test_acc:.4f}')
plt.figure(figsize=(12, 5))

# 绘制训练准确率曲线
plt.subplot(1, 2, 1)
plt.plot(range(1, epochs + 1), cnn_train_accs, label='Complex CNN (Train)', color='blue')
plt.plot(range(1, epochs + 1), fc_train_accs, label='FC (Train)', color='red')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Train Accuracy Comparison')
plt.legend()
plt.grid(True)
# 绘制测试准确率曲线
plt.subplot(1, 2, 2)
plt.plot(range(1, epochs + 1), cnn_test_accs, label='Complex CNN (Test)', color='blue')
plt.plot(range(1, epochs + 1), fc_test_accs, label='FC (Test)', color='red')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Test Accuracy Comparison')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()