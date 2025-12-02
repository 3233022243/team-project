import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchsummary import summary
import matplotlib.pyplot as plt


class Inception(nn.Module):
    def __init__(self, in_channels, ch1x1, ch3x3red, ch3x3, ch5x5red, ch5x5, pool_proj):
        super(Inception, self).__init__()
        self.branch1 = nn.Conv2d(in_channels, ch1x1, kernel_size=1)
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, ch3x3red, kernel_size=1),
            nn.ReLU(True),
            nn.Conv2d(ch3x3red, ch3x3, kernel_size=3, padding=1)
        )
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, ch5x5red, kernel_size=1),
            nn.ReLU(True),
            nn.Conv2d(ch5x5red, ch5x5, kernel_size=5, padding=2)
        )
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels, pool_proj, kernel_size=1)
        )

    def forward(self, x):
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)
        outputs = [branch1, branch2, branch3, branch4]
        return torch.cat(outputs, 1)


class GoogLeNetMNIST(nn.Module):
    def __init__(self, num_classes=10):
        super(GoogLeNetMNIST, self).__init__()
        self.pre_layers = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2)
        )
        self.a3 = Inception(32, 16, 16, 24, 4, 8, 8)
        self.b3 = Inception(56, 24, 24, 32, 8, 16, 16)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(88, num_classes)

    def forward(self, x):
        x = self.pre_layers(x)
        x = self.a3(x)
        x = self.b3(x)
        x = self.avgpool(x)
        x = self.dropout(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


def train(model, train_loader, test_loader, criterion, optimizer, device, epochs=10):
    # 记录训练过程数据
    train_losses = []
    train_accs = []
    test_losses = []
    test_accs = []

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0

        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()

        train_acc = 100. * correct / total
        train_losses.append(train_loss / len(train_loader))
        train_accs.append(train_acc)

        # 测试阶段
        test_loss = 0.0
        correct = 0
        total = 0
        model.eval()

        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                loss = criterion(output, target)

                test_loss += loss.item()
                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()

        test_acc = 100. * correct / total
        test_losses.append(test_loss / len(test_loader))
        test_accs.append(test_acc)

        print(
            f'Epoch {epoch + 1}, Train Loss: {train_losses[-1]:.4f}, Train Acc: {train_acc:.2f}%, Test Loss: {test_losses[-1]:.4f}, Test Acc: {test_acc:.2f}%')

    # 绘制训练曲线
    plt.figure(figsize=(12, 4))

    # 损失曲线
    plt.subplot(1, 2, 1)
    plt.plot(range(1, epochs + 1), train_losses, 'b-', label='Train Loss')
    plt.plot(range(1, epochs + 1), test_losses, 'r-', label='Test Loss')
    plt.title('Loss vs Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    # 准确率曲线
    plt.subplot(1, 2, 2)
    plt.plot(range(1, epochs + 1), train_accs, 'b-', label='Train Accuracy')
    plt.plot(range(1, epochs + 1), test_accs, 'r-', label='Test Accuracy')
    plt.title('Accuracy vs Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('googlenet_mnist_training_curve.png')
    plt.show()

    # 返回训练记录
    return {
        'train_losses': train_losses,
        'train_accs': train_accs,
        'test_losses': test_losses,
        'test_accs': test_accs
    }


if __name__ == "__main__":
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GoogLeNetMNIST().to(device)

    summary(model, (1, 28, 28))

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 训练并获取记录
    training_history = train(model, train_loader, test_loader, criterion, optimizer, device, epochs=10)

    # 打印最终结果
    print("\nFinal Results:")
    print(f"Final Train Accuracy: {training_history['train_accs'][-1]:.2f}%")
    print(f"Final Test Accuracy: {training_history['test_accs'][-1]:.2f}%")
    print(f"Best Test Accuracy: {max(training_history['test_accs']):.2f}%")