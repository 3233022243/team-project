import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import os

# 解决OpenMP库冲突（避免绘图时报错）
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# 1. 固定参数与数据预处理
idx2char = ['a', 'd', 'e', 'l', 'n', 'r']  # 字符→索引映射（按字母排序）
num_class = len(idx2char)  # 字符类别数（6类）
seq_len = 6
input_size = num_class
hidden_size = 8
batch_size = 1
learning_rate = 0.1
epochs = 30

# 输入输出序列
x_data = [1, 3, 2, 0, 5, 4]  # "dlearn"的索引
y_data = [3, 0, 4, 5, 3, 0]  # "lanrla"的索引

# One-Hot编码（适配RNN输入格式：[seq_len, batch_size, input_size]）
one_hot_lookup = torch.eye(num_class)
x_one_hot = one_hot_lookup[x_data].view(seq_len, batch_size, input_size)  # 形状：[6,1,6]
labels = torch.LongTensor(y_data)


# 2. RNN模型定义
class RNNSequenceModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_class):
        super(RNNSequenceModel, self).__init__()
        self.hidden_size = hidden_size
        self.rnn = nn.RNN(
            input_size=input_size,  # 输入特征维度
            hidden_size=hidden_size,
            batch_first=False
        )
        self.relu = nn.ReLU()
        self.fc = nn.Linear(
            in_features=hidden_size,
            out_features=num_class
        )

    def forward(self, x):
        # 初始化隐藏态（[num_layers, batch_size, hidden_size]）
        hidden = torch.zeros(1, x.size(1), self.hidden_size)
        # RNN前向传播 → [seq_len, batch_size, hidden_size]
        x, _ = self.rnn(x, hidden)
        x = self.relu(x)
        x = x.view(-1, self.hidden_size)
        x = self.fc(x)
        return x


# 3. 模型初始化与训练
net = RNNSequenceModel(
    input_size=input_size,
    hidden_size=hidden_size,
    num_class=num_class
)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=learning_rate)

# 4. 训练循环
print("开始训练（目标序列：lanrla）：")
loss_history = []  # 记录损失变化
for epoch in range(epochs):
    optimizer.zero_grad()  # 梯度清零
    outputs = net(x_one_hot)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()
    loss_history.append(loss.item())

    # 预测结果处理
    _, idx = outputs.max(dim=1)
    pred_str = ''.join([idx2char[x.item()] for x in idx])
    print(f'Epoch [{epoch + 1:2d}/{epochs}], Predicted: {pred_str:6s}, Loss: {loss.item():.3f}')

# 5. 生成损失率变化折线图
plt.figure(figsize=(8, 4))
plt.plot(range(1, len(loss_history) + 1), loss_history, label="RNN Loss", color="pink", linewidth=2)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('RNN Training Loss Curve (dlearn→lanrla)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()