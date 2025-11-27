import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import re
from collections import Counter
from sklearn.model_selection import train_test_split
import os

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 128
HIDDEN_SIZE = 64
N_LAYERS = 2
N_EPOCHS = 20
LEARNING_RATE = 0.001
MAX_VOCAB_SIZE = 5000
MAX_SEQ_LENGTH = 30
EMBEDDING_DIM = 64

DATA_PATH = r"F:\迅雷下载\train.tsv\train.tsv"

class MovieReviewDataset(Dataset):
    def __init__(self, phrases, sentiments, vocab_dict, max_seq_len):
        self.phrases = phrases
        self.sentiments = sentiments
        self.vocab_dict = vocab_dict
        self.max_seq_len = max_seq_len

    def __len__(self):
        return len(self.phrases)

    def __getitem__(self, idx):
        phrase = self.phrases[idx]
        seq = self.text_to_seq(phrase)

        if len(seq) == 0:
            seq = [1]  # <OOV>
        seq = self.pad_or_truncate(seq)
        sentiment = self.sentiments[idx]
        return torch.LongTensor(seq), torch.LongTensor([sentiment])

    def text_to_seq(self, text):
        text = re.sub(r'[^\w\s]', '', text.lower().strip())

        if not text:
            return []
        words = text.split()
        return [self.vocab_dict.get(word, 1) for word in words]

    def pad_or_truncate(self, seq):
        if len(seq) < self.max_seq_len:
            seq += [0] * (self.max_seq_len - len(seq))
        else:
            seq = seq[:self.max_seq_len]
        return seq

def build_vocab(phrases, max_vocab_size):
    all_words = []
    for phrase in phrases:
        phrase = re.sub(r'[^\w\s]', '', phrase.lower().strip())
        if phrase:
            all_words.extend(phrase.split())
    word_counter = Counter(all_words)
    common_words = word_counter.most_common(max_vocab_size - 2)
    vocab_dict = {'<PAD>': 0, '<OOV>': 1}
    for idx, (word, _) in enumerate(common_words, 2):
        vocab_dict[word] = idx
    return vocab_dict

def load_data(data_path):
    if not os.path.exists(data_path):
        print(f"错误：找不到文件 {data_path}")
        print("请检查文件路径是否正确！")
        exit(1)

    try:
        df = pd.read_csv(data_path, sep='\t')
        print(f"成功加载数据，共{len(df)}条记录")
    except Exception as e:
        print(f"加载数据失败：{str(e)}")
        exit(1)

    df = df.drop_duplicates(subset=['Phrase']).reset_index(drop=True)
    df['Phrase'] = df['Phrase'].str.strip()
    df = df[df['Phrase'].notna() & (df['Phrase'] != '')].reset_index(drop=True)
    print(f"过滤空数据后，剩余{len(df)}条记录")

    phrases = df['Phrase'].values
    sentiments = df['Sentiment'].values

    train_phrases, test_phrases, train_sents, test_sents = train_test_split(
        phrases, sentiments, test_size=0.2, random_state=42, stratify=sentiments
    )

    vocab_dict = build_vocab(train_phrases, MAX_VOCAB_SIZE)
    train_dataset = MovieReviewDataset(train_phrases, train_sents, vocab_dict, MAX_SEQ_LENGTH)
    test_dataset = MovieReviewDataset(test_phrases, test_sents, vocab_dict, MAX_SEQ_LENGTH)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    return train_loader, test_loader, vocab_dict

class TextRNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_size, output_size, n_layers, bidirectional=True):
        super(TextRNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.gru = nn.GRU(
            embed_dim, hidden_size, n_layers,
            bidirectional=bidirectional, batch_first=True, dropout=0.2
        )
        self.n_directions = 2 if bidirectional else 1
        self.fc = nn.Linear(hidden_size * self.n_directions, output_size)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x, seq_lengths):
        embed = self.embedding(x)
        embed = self.dropout(embed)

        seq_lengths = torch.clamp(seq_lengths, min=1)
        packed_embed = nn.utils.rnn.pack_padded_sequence(
            embed, seq_lengths.cpu(), batch_first=True, enforce_sorted=False
        )

        packed_output, hidden = self.gru(packed_embed)
        hidden = torch.cat([hidden[-1], hidden[-2]], dim=1) if self.n_directions == 2 else hidden[-1]
        hidden = self.dropout(hidden)
        out = self.fc(hidden)
        return out


# 计算序列真实长度（排除PAD）
def get_seq_lengths(seq):
    lengths = torch.sum(seq != 0, dim=1)
    # 确保长度至少为1
    lengths = torch.clamp(lengths, min=1)
    return lengths


# -------------------------- 4. 训练和测试函数 --------------------------
def train_model(model, train_loader, criterion, optimizer, epoch, start_time):
    model.train()
    total_loss = 0
    for batch_idx, (data, target) in enumerate(train_loader, 1):
        data, target = data.to(DEVICE), target.squeeze().to(DEVICE)
        seq_lengths = get_seq_lengths(data)

        output = model(data, seq_lengths)
        loss = criterion(output, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        if batch_idx % 20 == 0:
            elapsed = time_since(start_time)
            avg_loss = total_loss / (batch_idx * BATCH_SIZE)
            print(f'[{elapsed}] Epoch {epoch:2d} | Batch {batch_idx:3d} | Loss: {avg_loss:.4f}')

    return total_loss / len(train_loader.dataset)

def test_model(model, test_loader, criterion):
    model.eval()
    total_loss = 0
    correct = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(DEVICE), target.squeeze().to(DEVICE)
            seq_lengths = get_seq_lengths(data)

            output = model(data, seq_lengths)
            total_loss += criterion(output, target).item()

            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    avg_loss = total_loss / len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    print(f'\nTest Set | Average Loss: {avg_loss:.4f} | Accuracy: {accuracy:.2f}%\n')

    return accuracy


def time_since(since):
    s = time.time() - since
    m = int(s // 60)
    s -= m * 60
    return f'{m}m {s:.0f}s'


# -------------------------- 5. 主函数 --------------------------
if __name__ == '__main__':
    print("开始加载数据...")
    train_loader, test_loader, vocab_dict = load_data(DATA_PATH)

    VOCAB_SIZE = len(vocab_dict)
    OUTPUT_SIZE = 5

    print(f"\n词汇表大小: {VOCAB_SIZE}")
    print(f"训练集样本数: {len(train_loader.dataset)}")
    print(f"测试集样本数: {len(test_loader.dataset)}")
    print(f"使用设备: {DEVICE}")

    model = TextRNN(VOCAB_SIZE, EMBEDDING_DIM, HIDDEN_SIZE, OUTPUT_SIZE, N_LAYERS).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=2, factor=0.5)

    acc_list = []
    start_time = time.time()

    print("\n开始训练模型...")
    for epoch in range(1, N_EPOCHS + 1):
        train_loss = train_model(model, train_loader, criterion, optimizer, epoch, start_time)
        test_acc = test_model(model, test_loader, criterion)
        acc_list.append(test_acc)

        scheduler.step(test_acc)
        if epoch > 4 and max(acc_list[-4:]) <= max(acc_list[:-4]) + 0.3:
            print(f"早停触发，在Epoch {epoch}停止训练")
            break

    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(acc_list) + 1), acc_list, 'b-o', linewidth=2, markersize=8)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Test Accuracy (%)', fontsize=12)
    plt.title('TextRNN Sentiment Analysis Accuracy', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.xticks(range(1, len(acc_list) + 1))
    plt.show()

    torch.save({
        'model_state_dict': model.state_dict(),
        'vocab_dict': vocab_dict,
        'max_seq_length': MAX_SEQ_LENGTH,
        'embedding_dim': EMBEDDING_DIM,
        'hidden_size': HIDDEN_SIZE,
        'n_layers': N_LAYERS
    }, 'sentiment_analysis_model.pth')
    print("模型已保存为: sentiment_analysis_model.pth")

    def predict_sentiment(text):
        model.eval()
        text = re.sub(r'[^\w\s]', '', text.lower().strip())
        if not text:
            words = []
        else:
            words = text.split()
        seq = [vocab_dict.get(word, 1) for word in words]
        if len(seq) == 0:
            seq = [1]
        seq = seq[:MAX_SEQ_LENGTH] if len(seq) > MAX_SEQ_LENGTH else seq + [0] * (MAX_SEQ_LENGTH - len(seq))

        seq_tensor = torch.LongTensor([seq]).to(DEVICE)
        seq_length = torch.LongTensor([min(len(words), MAX_SEQ_LENGTH) if words else 1])

        with torch.no_grad():
            output = model(seq_tensor, seq_length)
            pred = output.argmax(dim=1).item()

        sentiment_map = {0: '消极', 1: '略带消极', 2: '中性', 3: '略带积极', 4: '积极'}
        return sentiment_map[pred]

    print("\n=== 情感预测示例 ===")
    test_examples = [
        "This movie is absolutely fantastic! I loved every minute of it.",
        "Terrible film, waste of money and time.",
        "It's an okay movie, not great but not bad either.",
        "The acting was good but the plot was boring and predictable.",
        "One of the best movies I've ever seen!",
        "",
        "   "
    ]

    for example in test_examples:
        sentiment = predict_sentiment(example)
        print(f"文本: '{example}'")
        print(f"预测情感: {sentiment}\n")