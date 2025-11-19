import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # 添加这行代码

# 后续正常导入其他库
import torch
import matplotlib.pyplot as plt
import numpy as np

# ... 其余代码不变 ...

# 构建字符映射字典
idx2char = ['d', 'l', 'e', 'a', 'r', 'n']  # 所有涉及的字符
char2idx = {char: idx for idx, char in enumerate(idx2char)}
input_size = len(idx2char)  # 输入维度=字符种类数=6
hidden_size = 6  # 隐藏层维度，可自定义
batch_size = 1  # 批次大小
seq_len = len("dlearn")  # 序列长度=6

# 训练数据：输入"dlearn"，输出"lanrla"
x_data = [char2idx[char] for char in "dlearn"]  # 输入索引：d(0), l(1), e(2), a(3), r(4), n(5)
y_data = [char2idx[char] for char in "lanrla"]  # 输出索引：l(1), a(3), n(5), r(4), l(1), a(3)

# 构建one-hot编码（RNNCell输入需为one-hot向量）
one_hot_lookup = torch.eye(input_size)  # 6x6的单位矩阵
x_one_hot = one_hot_lookup[x_data]  # 转换为one-hot向量，shape=(6,6)
# 调整输入格式：(seq_len, batch_size, input_size)
inputs = x_one_hot.view(seq_len, batch_size, input_size)
# 调整标签格式：(seq_len, 1)
labels = torch.LongTensor(y_data).view(-1, 1)


class RNNCellModel(torch.nn.Module):
    def __init__(self, input_size, hidden_size, batch_size):
        super(RNNCellModel, self).__init__()
        self.batch_size = batch_size
        self.input_size = input_size
        self.hidden_size = hidden_size
        # 定义RNNCell
        self.rnncell = torch.nn.RNNCell(input_size=input_size, hidden_size=hidden_size)

    def forward(self, input, hidden):
        # 前向传播：单个时间步计算
        hidden = self.rnncell(input, hidden)
        return hidden

    def init_hidden(self):
        # 初始化隐藏层为全0
        return torch.zeros(self.batch_size, self.hidden_size)


# 实例化模型
net_cell = RNNCellModel(input_size, hidden_size, batch_size)

# 损失函数与优化器
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net_cell.parameters(), lr=0.1)

# 记录训练过程：epoch、loss、预测结果
train_history_cell = []

# 训练15轮
for epoch in range(15):
    loss = 0.0
    optimizer.zero_grad()
    hidden = net_cell.init_hidden()  # 初始化隐藏层
    predicted_chars = []  # 记录当前轮预测字符

    # 逐时间步处理序列
    for input_t, label_t in zip(inputs, labels):
        hidden = net_cell(input_t, hidden)  # 前向传播
        loss += criterion(hidden, label_t)  # 累积损失
        # 预测当前时间步字符（取概率最大的索引）
        _, idx = hidden.max(dim=1)
        predicted_chars.append(idx2char[idx.item()])

    # 反向传播与参数更新
    loss.backward()
    optimizer.step()

    # 记录训练数据
    predicted_str = ''.join(predicted_chars)
    train_history_cell.append({
        "epoch": epoch + 1,
        "loss": loss.item(),
        "predicted": predicted_str
    })

    # 打印训练信息
    print(f'RNNCell - Epoch [{epoch + 1}/15], Loss=%.4f' % loss.item(),
          f'Predicted: {predicted_str}')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 提取数据（仅使用RNNCell的损失）
epochs = np.arange(1, 16)
loss_cell = [h["loss"] for h in train_history_cell]

# 绘图（仅绘制RNNCell的Loss曲线）
plt.figure(figsize=(10, 6))
plt.plot(epochs, loss_cell, label='RNNCell', marker='o', color='blue', linewidth=2)

# 图表样式
plt.xlabel('训练轮次（Epoch）', fontsize=12)
plt.ylabel('损失值（Loss）', fontsize=12)
plt.title('RNNCell的Loss变化曲线', fontsize=14)
plt.legend()
plt.grid(alpha=0.3)
plt.xticks(epochs)
plt.tight_layout()

# 保存并显示图片
plt.savefig('rnncell_loss曲线.png', dpi=300)
plt.show()

# 最终结果打印
print("\n===== 最终训练结果 =====")
print(f"RNNCell最后预测: {train_history_cell[-1]['predicted']}")
print(f"目标输出: lanrla")
