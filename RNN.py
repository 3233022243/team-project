import os

# 解决OpenMP冲突问题（如无此错误可注释）
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import matplotlib.pyplot as plt
import numpy as np

# ----------------------------
# 1. 数据准备
# ----------------------------
# 字符映射字典
idx2char = ['d', 'l', 'e', 'a', 'r', 'n']  # 所有涉及的字符
char2idx = {char: idx for idx, char in enumerate(idx2char)}
input_size = len(idx2char)  # 输入维度=6（字符种类数）
hidden_size = 6  # 隐藏层维度
batch_size = 1  # 批次大小
seq_len = len("dlearn")  # 序列长度=6

# 训练数据：输入"dlearn"，输出"lanrla"
x_data = [char2idx[char] for char in "dlearn"]  # 输入索引：d(0),l(1),e(2),a(3),r(4),n(5)
y_data = [char2idx[char] for char in "lanrla"]  # 输出索引：l(1),a(3),n(5),r(4),l(1),a(3)

# 构建one-hot编码（RNN输入需为one-hot向量）
one_hot_lookup = torch.eye(input_size)  # 6x6单位矩阵
x_one_hot = one_hot_lookup[x_data]  # 转换为one-hot向量，shape=(6,6)

# 调整输入输出格式
inputs = x_one_hot.view(seq_len, batch_size, input_size)  # RNN输入格式：(seq_len, batch_size, input_size)
labels = torch.LongTensor(y_data)  # RNN标签格式：(seq_len,)


# ----------------------------
# 2. 定义RNN模型
# ----------------------------
class RNNModel(torch.nn.Module):
    def __init__(self, input_size, hidden_size, batch_size, num_layers=1):
        super(RNNModel, self).__init__()
        self.num_layers = num_layers  # RNN层数（此处为1）
        self.batch_size = batch_size
        self.hidden_size = hidden_size
        # 定义RNN层（nn.RNN自动处理整个序列）
        self.rnn = torch.nn.RNN(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=False  # 输入格式为(seq_len, batch_size, input_size)
        )

    def forward(self, input):
        # 初始化隐藏层：(num_layers, batch_size, hidden_size)
        hidden = torch.zeros(self.num_layers, self.batch_size, self.hidden_size)
        # 前向传播：out为所有时间步的输出，hidden为最后一个时间步的隐藏状态
        out, _ = self.rnn(input, hidden)
        # 调整输出格式以匹配损失函数：(seq_len * batch_size, hidden_size)
        return out.view(-1, self.hidden_size)


# ----------------------------
# 3. 模型训练
# ----------------------------
# 实例化模型
net_rnn = RNNModel(input_size, hidden_size, batch_size)

# 损失函数与优化器
criterion = torch.nn.CrossEntropyLoss()  # 适用于分类问题
optimizer = torch.optim.Adam(net_rnn.parameters(), lr=0.05)  # 优化器

# 记录训练过程
train_history_rnn = []

# 训练15轮
for epoch in range(15):
    optimizer.zero_grad()  # 清空梯度
    outputs = net_rnn(inputs)  # 前向传播：一次性处理整个序列
    loss = criterion(outputs, labels)  # 计算损失

    # 反向传播与参数更新
    loss.backward()
    optimizer.step()

    # 预测当前轮结果
    _, idx = outputs.max(dim=1)  # 取概率最大的索引
    predicted_chars = [idx2char[i.item()] for i in idx]
    predicted_str = ''.join(predicted_chars)

    # 记录训练数据
    train_history_rnn.append({
        "epoch": epoch + 1,
        "loss": loss.item(),
        "predicted": predicted_str
    })

    # 打印训练信息
    print(f'RNN - Epoch [{epoch + 1}/15], Loss={loss.item():.4f}, 预测: {predicted_str}')

# ----------------------------
# 4. Loss可视化
# ----------------------------
# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 提取数据
epochs = np.arange(1, 16)  # 1-15轮
loss_rnn = [h["loss"] for h in train_history_rnn]  # RNN的loss数据

# 绘图
plt.figure(figsize=(10, 6))
plt.plot(epochs, loss_rnn, label='RNN', marker='s', color='orange', linewidth=2)

# 图表样式
plt.xlabel('训练轮次（Epoch）', fontsize=12)
plt.ylabel('损失值（Loss）', fontsize=12)
plt.title('RNN的Loss变化曲线', fontsize=14)
plt.legend()
plt.grid(alpha=0.3)
plt.xticks(epochs)  # 显示每一轮次
plt.tight_layout()

# 保存并显示图片
plt.savefig('rnn_loss曲线.png', dpi=300)
plt.show()

# ----------------------------
# 5. 最终结果打印
# ----------------------------
print("\n===== 最终训练结果 =====")
print(f"RNN最后预测: {train_history_rnn[-1]['predicted']}")
print(f"目标输出: lanrla")