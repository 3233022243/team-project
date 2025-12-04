import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']  # 中文显示
plt.rcParams['axes.unicode_minus'] = False  # 负号显示
# 设备设置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

# 数据预处理（GoogLeNet默认输入224x224，适配MNIST调整为64x64）
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.Grayscale(num_output_channels=3),  # GoogLeNet需要3通道
    transforms.ToTensor(),
    transforms.Normalize((0.1307, 0.1307, 0.1307), (0.3081, 0.3081, 0.3081))
])

# 加载MNIST数据集
train_dataset = datasets.MNIST(root=r"./mnist_data", train=True, transform=transform, download=False)
test_dataset = datasets.MNIST(root=r"./mnist_data", train=False, transform=transform, download=False)

train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)


# 定义GoogLeNet
class Inception(nn.Module):
    def __init__(self, in_channels, ch1x1, ch3x3red, ch3x3, ch5x5red, ch5x5, pool_proj):
        super(Inception, self).__init__()
        # 1x1卷积
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, ch1x1, kernel_size=1),
            nn.BatchNorm2d(ch1x1),
            nn.ReLU(inplace=True)
        )
        # 3x3卷积（先1x1降维）
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, ch3x3red, kernel_size=1),
            nn.BatchNorm2d(ch3x3red),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch3x3red, ch3x3, kernel_size=3, padding=1),
            nn.BatchNorm2d(ch3x3),
            nn.ReLU(inplace=True)
        )
        # 5x5卷积（先1x1降维）
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, ch5x5red, kernel_size=1),
            nn.BatchNorm2d(ch5x5red),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch5x5red, ch5x5, kernel_size=5, padding=2),
            nn.BatchNorm2d(ch5x5),
            nn.ReLU(inplace=True)
        )
        # 池化+1x1卷积
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels, pool_proj, kernel_size=1),
            nn.BatchNorm2d(pool_proj),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)
        outputs = [branch1, branch2, branch3, branch4]
        return torch.cat(outputs, 1)

class GoogLeNet(nn.Module):
    def __init__(self, num_classes=10):
        super(GoogLeNet, self).__init__()
        self.pre_layers = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        self.a3 = Inception(64, 64, 96, 128, 16, 32, 32)
        self.b3 = Inception(128 + 32 + 32 + 64, 128, 128, 192, 32, 96, 64)  # 64+128+32+32=256
        self.maxpool3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.a4 = Inception(192 + 96 + 64 + 128, 192, 96, 208, 16, 48, 64)  # 128+192+96+64=480
        self.b4 = Inception(208 + 48 + 64 + 192, 160, 112, 224, 24, 64, 64)  # 192+208+48+64=512
        self.c4 = Inception(224 + 64 + 64 + 160, 128, 128, 256, 24, 64, 64)  # 160+224+64+64=480
        self.d4 = Inception(256 + 64 + 64 + 128, 112, 144, 288, 32, 64, 64)  # 128+256+64+64=512
        self.e4 = Inception(288 + 64 + 64 + 112, 256, 160, 320, 32, 128, 128)  # 112+288+64+64=528
        self.maxpool4 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.a5 = Inception(320 + 128 + 128 + 256, 256, 160, 320, 32, 128, 128)  # 256+320+128+128=832
        self.b5 = Inception(320 + 128 + 128 + 256, 384, 192, 384, 48, 128, 128)  # 256+320+128+128=832
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.4)
        self.fc = nn.Linear(384 + 128 + 128 + 384, num_classes)  # 384+384+128+128=102
    def forward(self, x):
        x = self.pre_layers(x)
        x = self.a3(x)
        x = self.b3(x)
        x = self.maxpool3(x)
        x = self.a4(x)
        x = self.b4(x)
        x = self.c4(x)
        x = self.d4(x)
        x = self.e4(x)
        x = self.maxpool4(x)
        x = self.a5(x)
        x = self.b5(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        return x


# 初始化模型、损失、优化器
model = GoogLeNet(num_classes=10).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-5)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)

# 训练
num_epochs = 5
train_losses = []
train_acc = []
test_acc = []

print("开始训练GoogLeNet...")
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, pred = output.max(1)
        total += target.size(0)
        correct += pred.eq(target).sum().item()

        if batch_idx % 100 == 99:
            print(
                f"Epoch [{epoch + 1}/{num_epochs}], Step [{batch_idx + 1}/{len(train_loader)}], Loss: {loss.item():.4f}")

    # 记录训练指标
    epoch_loss = running_loss / len(train_loader)
    epoch_train_acc = correct / total
    train_losses.append(epoch_loss)
    train_acc.append(epoch_train_acc)

    # 测试
    model.eval()
    test_correct = 0
    test_total = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, pred = output.max(1)
            test_total += target.size(0)
            test_correct += pred.eq(target).sum().item()
    epoch_test_acc = test_correct / test_total
    test_acc.append(epoch_test_acc)

    scheduler.step()
    print(
        f"Epoch [{epoch + 1}/{num_epochs}], 训练损失: {epoch_loss:.4f}, 训练准确率: {epoch_train_acc:.4f}, 测试准确率: {epoch_test_acc:.4f}")

# 性能评估
model.eval()
all_preds = []
all_targets = []
with torch.no_grad():
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        output = model(data)
        _, pred = output.max(1)
        all_preds.extend(pred.cpu().numpy())
        all_targets.extend(target.cpu().numpy())

# 计算指标
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
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(range(1, num_epochs + 1), train_losses, 'b-', label='训练损失')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('GoogLeNet训练损失曲线')
plt.legend()
plt.grid()

plt.subplot(1, 2, 2)
plt.plot(range(1, num_epochs + 1), train_acc, 'r-', label='训练准确率')
plt.plot(range(1, num_epochs + 1), test_acc, 'g--', label='测试准确率')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('GoogLeNet准确率曲线')
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()


# 手写数字预测函数
def predict_digit(image_path):
    from PIL import Image
    img = Image.open(image_path).convert('L')
    img = img.resize((64, 64))
    # 自动反转黑白
    if np.array(img).mean() > 127:
        img = Image.eval(img, lambda x: 255 - x)
    img = transforms.Grayscale(num_output_channels=3)(img)
    img_tensor = transform(img).unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        output = model(img_tensor)
        prob = torch.softmax(output, dim=1)
        pred = output.argmax(1).item()
        confidence = prob[0][pred].item()

    # 显示结果
    plt.figure(figsize=(6, 4))
    plt.imshow(img.convert('L'), cmap='gray')
    plt.title(f'预测数字: {pred}\n置信度: {confidence:.2%}')
    plt.axis('off')
    plt.show()
    return pred, confidence


# 测试预测
try:
    pred, conf = predict_digit("C:/Users/15198/PycharmProjects/PythonProject/images/eight.jpg")
    print(f"\n预测结果: {pred} (置信度: {conf:.2%})")
except Exception as e:
    print(f"预测出错: {e}")