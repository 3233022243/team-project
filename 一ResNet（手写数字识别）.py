import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from PIL import Image

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 设备设置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

# 数据预处理
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.Grayscale(num_output_channels=3),  # ResNet需要3通道输入
    transforms.ToTensor(),
    transforms.Normalize((0.1307, 0.1307, 0.1307), (0.3081, 0.3081, 0.3081))
])

# 加载MNIST数据集
train_dataset = datasets.MNIST(root=r"./mnist_data", train=True, transform=transform, download=False)
test_dataset = datasets.MNIST(root=r"./mnist_data", train=False, transform=transform, download=False)

train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)


# 定义ResNet的残差块
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != self.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, self.expansion * out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * out_channels)
            )

    def forward(self, x):
        out = nn.ReLU()(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = nn.ReLU()(out)
        return out


# 定义ResNet网络
class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_channels = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = nn.ReLU()(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)
        return out


# 创建ResNet18模型
def ResNet18():
    return ResNet(BasicBlock, [2, 2, 2, 2])


# 初始化模型、损失函数和优化器
model = ResNet18().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)

# 训练模型
num_epochs = 5
train_losses = []
train_accuracies = []
test_accuracies = []

print("\n开始训练ResNet18...")
for epoch in range(num_epochs):
    # 训练阶段
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (data, targets) in enumerate(train_loader):
        data, targets = data.to(device), targets.to(device)

        # 前向传播
        outputs = model(data)
        loss = criterion(outputs, targets)

        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        if batch_idx % 100 == 99:
            print(
                f'Epoch [{epoch + 1}/{num_epochs}], Step [{batch_idx + 1}/{len(train_loader)}], Loss: {loss.item():.4f}')

    # 记录训练指标
    epoch_loss = running_loss / len(train_loader)
    epoch_train_acc = correct / total
    train_losses.append(epoch_loss)
    train_accuracies.append(epoch_train_acc)

    # 测试阶段
    model.eval()
    test_correct = 0
    test_total = 0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for data, targets in test_loader:
            data, targets = data.to(device), targets.to(device)
            outputs = model(data)
            _, predicted = outputs.max(1)
            test_total += targets.size(0)
            test_correct += predicted.eq(targets).sum().item()

            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

    epoch_test_acc = test_correct / test_total
    test_accuracies.append(epoch_test_acc)

    # 更新学习率
    scheduler.step()

    print(f'Epoch [{epoch + 1}/{num_epochs}]')
    print(f'训练损失: {epoch_loss:.4f}, 训练准确率: {epoch_train_acc:.4f}')
    print(f'测试准确率: {epoch_test_acc:.4f}\n')

# 计算详细的性能指标
accuracy = accuracy_score(all_targets, all_preds)
precision = precision_score(all_targets, all_preds, average='weighted')
recall = recall_score(all_targets, all_preds, average='weighted')
f1 = f1_score(all_targets, all_preds, average='weighted')

print("\n测试集评估指标:")
print(f"准确率 (Accuracy): {accuracy:.4f}")
print(f"精确率 (Precision): {precision:.4f}")
print(f"召回率 (Recall): {recall:.4f}")
print(f"F1分数 (F1 Score): {f1:.4f}")

# 绘制训练曲线
plt.figure(figsize=(10, 5))

# 损失曲线
plt.subplot(1, 3, 1)
plt.plot(range(1, num_epochs + 1), train_losses, 'b-', marker='o', label='训练损失')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('训练损失曲线')
plt.legend()
plt.grid(True)

# 准确率曲线
plt.subplot(1, 3, 2)
plt.plot(range(1, num_epochs + 1), train_accuracies, 'r-', marker='o', label='训练准确率')
plt.plot(range(1, num_epochs + 1), test_accuracies, 'g-', marker='s', label='测试准确率')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('准确率曲线')
plt.legend()
plt.grid(True)



plt.tight_layout()
plt.show()


# 手写数字预测函数
def predict_handwritten_digit(image_path):
    try:
        # 加载并预处理图片
        img = Image.open(image_path).convert('L')
        img = img.resize((32, 32))

        # 自动调整黑白反转
        img_np = np.array(img)
        if img_np.mean() > 127:
            img = Image.eval(img, lambda x: 255 - x)

        # 转为3通道并应用预处理
        img = transforms.Grayscale(num_output_channels=3)(img)
        img_tensor = transform(img).unsqueeze(0).to(device)

        # 预测
        model.eval()
        with torch.no_grad():
            output = model(img_tensor)
            probabilities = torch.nn.functional.softmax(output, dim=1)
            _, predicted = torch.max(output, 1)
            predicted_digit = predicted.item()
            confidence = probabilities[0][predicted_digit].item()

        # 显示结果
        plt.figure(figsize=(10, 4))

        # 显示图片
        plt.subplot(1, 2, 1)
        plt.imshow(img.convert('L'), cmap='gray')
        plt.title(f'预测数字: {predicted_digit}\n置信度: {confidence:.2%}', fontsize=14)
        plt.axis('off')


        plt.tight_layout()
        plt.show()

        return predicted_digit, confidence

    except Exception as e:
        print(f"\n预测出错: {e}")
        return None, 0.0


# 测试预测功能
try:
    predicted_digit, confidence = predict_handwritten_digit(
        "C:/Users/15198/PycharmProjects/PythonProject/images/eight.jpg")
    if predicted_digit is not None:
        print(f"\n手写数字预测结果: {predicted_digit} (置信度: {confidence:.2%})")
except Exception as e:
    print(f"\n预测失败: {e}")
    print("请检查图片路径是否正确")




