import torch
import matplotlib.pyplot as plt
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# 1. 固定参数
input_size = 6
hidden_size = 8
batch_size = 1
idx2char = ['a', 'd', 'e', 'l', 'n', 'r']
x_data = [1, 3, 2, 0, 5, 4]
y_data = [3, 0, 4, 5, 3, 0]

# 2. One-Hot编码
one_hot_lookup = [
    [1, 0, 0, 0, 0, 0],
    [0, 1, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, 0],
    [0, 0, 0, 1, 0, 0],
    [0, 0, 0, 0, 1, 0],
    [0, 0, 0, 0, 0, 1]
]
x_one_hot = [one_hot_lookup[x] for x in x_data]
inputs = torch.Tensor(x_one_hot).view(-1, batch_size, input_size)
labels = torch.LongTensor(y_data).view(-1, 1)

# 3. 模型定义
class RNNCellModel(torch.nn.Module):
    def __init__(self, input_size, hidden_size, batch_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.rnncell = torch.nn.RNNCell(input_size=input_size, hidden_size=hidden_size)
        self.fc = torch.nn.Linear(hidden_size, input_size)

    def forward(self, input, hidden):
        hidden = self.rnncell(input, hidden)
        output = self.fc(hidden)
        return output, hidden

    def init_hidden(self):
        return torch.zeros(self.batch_size, self.hidden_size)

# 4. 训练
net = RNNCellModel(input_size, hidden_size, batch_size)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=0.1)
loss_history = []

for epoch in range(20):
    loss = 0.0
    optimizer.zero_grad()
    hidden = net.init_hidden()  # 现在不会报错
    print('Predicted string: ', end='')
    for input, label in zip(inputs, labels):
        output, hidden = net(input, hidden)
        loss += criterion(output, label)
        _, idx = output.max(dim=1)
        print(idx2char[idx.item()], end='')
    loss.backward()
    optimizer.step()
    loss_history.append(loss.item())
    print(f', Epoch [{epoch + 1:2d}/20] loss=%.4f' % (loss.item()))

# 5. 绘图
plt.figure(figsize=(8, 4))
plt.plot(range(1, 21), loss_history, label="RNNCell Loss", linewidth=2)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('RNNCell Training Loss Curve (dlearn→lanrla)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()