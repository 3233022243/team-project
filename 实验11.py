# 1. 环境依赖导入
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import os
from collections import Counter
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# 2. 全局参数配置（核心修改：EPOCHS=10，适配你的路径）
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 自动选择CPU/GPU
BATCH_SIZE = 256
EMBEDDING_DIM = 128  # 词嵌入维度
HIDDEN_SIZE = 100  # RNN隐藏层维度
NUM_LAYERS = 2  # RNN层数
DROPOUT = 0.5  # Dropout概率（防止过拟合）
LEARNING_RATE = 0.001  # 学习率
EPOCHS = 10  # 核心修改：训练轮次从50改为10
MAX_VOCAB_SIZE = 10000  # 最大词汇表大小（过滤低频词）
MAX_SEQ_LENGTH = 50  # 句子固定长度（截断/填充）
# 你的数据集路径（嵌套路径，若简化后可修改为下方注释的简化路径）
TRAIN_DATA_PATH = "D:\\人工智能\\大三\\神经网络\\train.tsv\\train.tsv"


# TRAIN_DATA_PATH = "D:\\人工智能\\大三\\神经网络\\train.tsv"  # 简化路径（若文件已移出嵌套文件夹）

# 3. 数据加载与预处理
def load_data(data_path):
    """加载烂番茄数据集，返回评论文本和情感标签列表"""
    # 增加encoding='utf-8'，避免中文路径导致的编码问题
    df = pd.read_csv(data_path, sep='\t', encoding='utf-8')
    texts = df['Phrase'].values.tolist()  # 提取评论文本
    labels = df['Sentiment'].values.tolist()  # 提取情感标签（0-消极 ~ 4-积极）
    return texts, labels


def build_vocab(texts, max_vocab_size):
    """基于训练集构建词汇表（含<PAD>填充符和<UNK>未登录词）"""
    word_counter = Counter()
    # 统计词频（小写处理，避免大小写重复计数）
    for text in texts:
        words = text.lower().split()
        word_counter.update(words)
    # 筛选高频词，预留2个位置给<PAD>（0）和<UNK>（1）
    common_words = word_counter.most_common(max_vocab_size - 2)
    vocab = {'<PAD>': 0, '<UNK>': 1}
    for idx, (word, _) in enumerate(common_words, 2):
        vocab[word] = idx
    return vocab


def text_to_tensor(text, vocab, max_seq_length):
    """将单条文本转换为固定长度的张量（词→索引→截断/填充）"""
    words = text.lower().split()
    # 词转索引：未登录词用<UNK>（索引1）替代
    indices = [vocab.get(word, vocab['<UNK>']) for word in words]
    # 调整长度：超过max_seq_length截断，不足则用<PAD>（索引0）填充
    if len(indices) > max_seq_length:
        indices = indices[:max_seq_length]
    else:
        indices += [vocab['<PAD>']] * (max_seq_length - len(indices))
    return torch.LongTensor(indices)


class MovieReviewDataset(Dataset):
    """自定义数据集类，适配PyTorch的DataLoader批量加载"""

    def __init__(self, texts, labels, vocab, max_seq_length):
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        self.max_seq_length = max_seq_length

    def __len__(self):
        """返回数据集总样本数"""
        return len(self.texts)

    def __getitem__(self, idx):
        """根据索引返回单条数据（文本张量+标签张量）"""
        # 文本编码为张量
        text_tensor = text_to_tensor(self.texts[idx], self.vocab, self.max_seq_length)
        # 标签转换为张量（挤压维度，适配交叉熵损失输入）
        label_tensor = torch.LongTensor([self.labels[idx]]).squeeze(0)
        return text_tensor, label_tensor


# 4. TextRNN模型定义（双向GRU，捕捉前后向上下文）
class TextRNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, num_layers, num_classes, dropout):
        super(TextRNN, self).__init__()
        # 嵌入层：将词索引转换为稠密向量（不更新<PAD>的嵌入）
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=0  # <PAD>索引0，训练时不更新其嵌入向量
        )
        # 双向GRU层（缓解梯度消失，比普通RNN训练更稳定）
        self.rnn = nn.GRU(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=True,  # 启用双向（前向+后向）
            batch_first=True,  # 输入格式：(batch_size, seq_length, embedding_dim)
            dropout=dropout if num_layers > 1 else 0  # 多层时启用中间层Dropout
        )
        # 全连接层：双向GRU输出需拼接（2*hidden_size），映射到5个情感类别
        self.fc = nn.Linear(hidden_size * 2, num_classes)
        # Dropout层：全连接前添加，进一步防止过拟合
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """前向传播流程：文本张量→嵌入→GRU→全连接→类别预测"""
        # 嵌入层：(batch_size, seq_length) → (batch_size, seq_length, embedding_dim)
        embed = self.embedding(x)
        embed = self.dropout(embed)  # 嵌入后加Dropout

        # GRU层：output=(batch_size, seq_length, 2*hidden_size)，hidden=(2*num_layers, batch_size, hidden_size)
        output, hidden = self.rnn(embed)

        # 取双向GRU最后一个时间步的隐藏层（前向最后一步+后向最后一步）
        last_hidden = torch.cat([hidden[-1], hidden[-2]], dim=1)  # (batch_size, 2*hidden_size)
        last_hidden = self.dropout(last_hidden)  # 全连接前加Dropout

        # 全连接层：输出类别logits（未经过softmax，交叉熵损失自动处理）
        logits = self.fc(last_hidden)  # (batch_size, num_classes)
        return logits


# 5. 训练与测试函数
def train_model(model, train_loader, criterion, optimizer, epoch):
    """单轮训练函数：更新模型参数，返回平均损失"""
    model.train()  # 切换为训练模式（启用Dropout、BatchNorm更新）
    total_loss = 0.0
    for batch_idx, (texts, labels) in enumerate(train_loader):
        # 数据移至指定设备（CPU/GPU）
        texts, labels = texts.to(DEVICE), labels.to(DEVICE)

        # 1. 前向传播：计算模型输出
        outputs = model(texts)
        # 2. 计算损失（多分类交叉熵损失）
        loss = criterion(outputs, labels)

        # 3. 反向传播与参数优化
        optimizer.zero_grad()  # 清空梯度（避免累积）
        loss.backward()  # 计算梯度
        optimizer.step()  # 更新模型参数

        # 累计损失
        total_loss += loss.item()

        # 每50个批次打印训练进度（避免输出过多）
        if (batch_idx + 1) % 50 == 0:
            avg_loss = total_loss / (batch_idx + 1)
            print(f"Epoch [{epoch + 1}/{EPOCHS}] | Batch [{batch_idx + 1}/{len(train_loader)}] | Loss: {avg_loss:.4f}")

    # 返回本轮平均损失
    return total_loss / len(train_loader)


def test_model(model, test_loader, criterion):
    """测试函数：评估模型性能（无梯度计算），返回平均损失和准确率"""
    model.eval()  # 切换为评估模式（禁用Dropout、固定BatchNorm）
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():  # 禁用梯度计算，节省内存和时间
        for texts, labels in test_loader:
            # 数据移至指定设备
            texts, labels = texts.to(DEVICE), labels.to(DEVICE)

            # 前向传播
            outputs = model(texts)
            # 计算测试损失
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            # 计算准确率：取logits最大值对应的索引作为预测类别
            _, predicted = torch.max(outputs.data, 1)  # predicted: (batch_size,)
            total += labels.size(0)  # 总样本数
            correct += (predicted == labels).sum().item()  # 正确预测数

    # 计算平均损失和准确率
    avg_loss = total_loss / len(test_loader)
    accuracy = 100 * correct / total
    print(f"\nTest Result | Loss: {avg_loss:.4f} | Accuracy: {accuracy:.2f}%\n")
    return avg_loss, accuracy


# 6. 主函数（完整流程入口，含路径验证和权限处理）
if __name__ == "__main__":
    # 第一步：验证数据集路径是否存在（避免FileNotFoundError）
    if not os.path.exists(TRAIN_DATA_PATH):
        print(f" 错误：文件不存在！当前路径：{TRAIN_DATA_PATH}")
        print("请检查：1. 路径中文件夹/文件名是否拼写正确 2. 文件是否在该路径下")
        exit()  # 路径错误时退出，避免后续报错

    # 第二步：加载数据（处理权限问题）
    print("=== 1. 加载并预处理数据 ===")
    try:
        texts, labels = load_data(TRAIN_DATA_PATH)
    except PermissionError:
        print("权限被拒绝！解决方案：")
        print("1. 以管理员身份运行代码编辑器；")
        print("2. 将train.tsv移动到桌面或新建文件夹（如D:\\MovieData），并修改TRAIN_DATA_PATH。")
        exit()

    # 第三步：划分训练集（80%）和测试集（20%）（固定random_state=42确保可复现）
    train_texts, test_texts, train_labels, test_labels = train_test_split(
        texts, labels, test_size=0.2, random_state=42
    )

    # 第四步：构建词汇表（仅基于训练集，避免数据泄露）
    vocab = build_vocab(train_texts, MAX_VOCAB_SIZE)
    vocab_size = len(vocab)
    print(f"词汇表大小：{vocab_size}（含<PAD>和<UNK>）")

    # 第五步：创建数据集和数据加载器（批量加载+训练集打乱）
    train_dataset = MovieReviewDataset(train_texts, train_labels, vocab, MAX_SEQ_LENGTH)
    test_dataset = MovieReviewDataset(test_texts, test_labels, vocab, MAX_SEQ_LENGTH)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    print(f"训练集批次数量：{len(train_loader)} | 测试集批次数量：{len(test_loader)}")

    # 第六步：初始化模型、损失函数和优化器
    print("\n=== 2. 初始化模型 ===")
    model = TextRNN(
        vocab_size=vocab_size,
        embedding_dim=EMBEDDING_DIM,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        num_classes=5,  # 情感标签共5类（0-4）
        dropout=DROPOUT
    ).to(DEVICE)  # 将模型移至指定设备（CPU/GPU）

    criterion = nn.CrossEntropyLoss()  # 多分类交叉熵损失
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)  # Adam优化器
    print(f"使用设备：{DEVICE} | 训练轮次：{EPOCHS}")

    # 第七步：记录训练指标（用于后续可视化）
    train_losses = []  # 训练损失
    test_losses = []  # 测试损失
    test_accuracies = []  # 测试准确率

    # 第八步：开始训练与测试循环（共10轮）
    print("\n=== 3. 开始训练（共10轮） ===")
    for epoch in range(EPOCHS):
        print(f"\n===== Epoch {epoch + 1}/{EPOCHS} =====")
        # 训练模型并记录损失
        train_loss = train_model(model, train_loader, criterion, optimizer, epoch)
        # 测试模型并记录损失和准确率
        test_loss, test_acc = test_model(model, test_loader, criterion)
        # 保存指标
        train_losses.append(train_loss)
        test_losses.append(test_loss)
        test_accuracies.append(test_acc)

    # 第九步：训练结果可视化（损失曲线+准确率曲线）
    print("\n=== 4. 训练结果可视化 ===")
    plt.figure(figsize=(12, 4))

    # 子图1：训练损失 vs 测试损失
    plt.subplot(1, 2, 1)
    plt.plot(range(1, EPOCHS + 1), train_losses, label="Train Loss", color="blue")
    plt.plot(range(1, EPOCHS + 1), test_losses, label="Test Loss", color="red")
    plt.xlabel("Epoch（训练轮次）")
    plt.ylabel("Loss（损失）")
    plt.legend()
    plt.title("Loss Curve（损失曲线）")
    plt.grid(alpha=0.3)  # 添加网格便于查看

    # 子图2：测试准确率曲线
    plt.subplot(1, 2, 2)
    plt.plot(range(1, EPOCHS + 1), test_accuracies, label="Test Accuracy", color="orange")
    plt.xlabel("Epoch（训练轮次）")
    plt.ylabel("Accuracy（准确率 %）")
    plt.legend()
    plt.title("Accuracy Curve（准确率曲线）")
    plt.grid(alpha=0.3)

    # 调整子图间距，显示图像
    plt.tight_layout()
    plt.show()

    # 第十步：保存训练好的模型（仅保存参数，便于后续推理）
    torch.save(model.state_dict(), "text_rnn_movie_review_10epochs.pth")
    print(f"\n 模型保存完成：text_rnn_movie_review_10epochs.pth")
    print("=== 10轮训练流程全部结束 ===")