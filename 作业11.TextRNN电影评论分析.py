import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import re
from collections import Counter
import time
# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']  # 中文显示
plt.rcParams['axes.unicode_minus'] = False  # 负号显示

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")


class Config:
    HIDDEN_SIZE = 128
    BATCH_SIZE = 64
    N_LAYER = 2
    N_EPOCHS = 5
    LEARNING_RATE = 0.001
    MAX_LEN = 100
    MIN_WORD_COUNT = 2


config = Config()


class TextProcessor:
    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.vocab_size = 0

    def clean_text(self, text):
        text = str(text).lower()
        text = re.sub(r"[^a-zA-Z.!?']", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def build_vocab(self, texts, min_count=2):
        word_counts = Counter()
        for text in texts:
            cleaned_text = self.clean_text(text)
            words = cleaned_text.split()
            word_counts.update(words)

        valid_words = [word for word, count in word_counts.items() if count >= min_count]

        self.word2idx = {'<PAD>': 0, '<UNK>': 1}
        self.idx2word = {0: '<PAD>', 1: '<UNK>'}

        for idx, word in enumerate(valid_words, 2):
            self.word2idx[word] = idx
            self.idx2word[idx] = word

        self.vocab_size = len(self.word2idx)
        print(f"词汇表大小: {self.vocab_size}")

    def text_to_sequence(self, text):
        cleaned_text = self.clean_text(text)
        words = cleaned_text.split()
        sequence = [self.word2idx.get(word, self.word2idx['<UNK>']) for word in words]
        return sequence


class MovieReviewDataset(Dataset):
    def __init__(self, texts, sentiments, text_processor, max_len=100):
        self.texts = texts
        self.sentiments = sentiments
        self.text_processor = text_processor
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        sentiment = self.sentiments[idx]
        sequence = self.text_processor.text_to_sequence(text)

        if len(sequence) > self.max_len:
            sequence = sequence[:self.max_len]
        else:
            sequence = sequence + [0] * (self.max_len - len(sequence))

        return torch.tensor(sequence, dtype=torch.long), torch.tensor(sentiment, dtype=torch.long)


class TextRNN(nn.Module):
    def __init__(self, vocab_size, hidden_size, output_size, n_layers=2, bidirectional=True):
        super(TextRNN, self).__init__()
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.n_directions = 2 if bidirectional else 1

        self.embedding = nn.Embedding(vocab_size, hidden_size, padding_idx=0)
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers, batch_first=True, bidirectional=bidirectional,
                          dropout=0.3)
        self.fc = nn.Linear(hidden_size * self.n_directions, output_size)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        embedded = self.embedding(x)
        gru_output, hidden = self.gru(embedded)

        if self.n_directions == 2:
            hidden = torch.cat([hidden[-1], hidden[-2]], dim=1)
        else:
            hidden = hidden[-1]

        output = self.fc(self.dropout(hidden))
        return output


def load_data(train_path, test_path):
    train_df = pd.read_csv(train_path, sep='\t')
    test_df = pd.read_csv(test_path, sep='\t')
    print(f"训练数据形状: {train_df.shape}")
    print(f"测试数据形状: {test_df.shape}")
    return train_df, test_df


def train_model(model, train_loader, criterion, optimizer, epoch):
    model.train()
    total_loss, correct, total = 0, 0, 0

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        total += target.size(0)

        if batch_idx % 50 == 0:
            print(f'训练轮次: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                  f'({100. * batch_idx / len(train_loader):.0f}%)]\t损失: {loss.item():.6f}')

    accuracy = 100. * correct / total
    avg_loss = total_loss / len(train_loader)
    return avg_loss, accuracy


def test_model(model, test_loader, criterion):
    model.eval()
    test_loss, correct, total = 0, 0, 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)

    test_loss /= len(test_loader)
    accuracy = 100. * correct / total
    print(f'\n测试集: 平均损失: {test_loss:.4f}, 准确率: {correct}/{total} ({accuracy:.2f}%)\n')
    return test_loss, accuracy


def main():
    train_path = r"D:\潘 +深度学习\train.tsv"
    test_path = r"D:\潘 +深度学习\test.tsv"

    train_df, test_df = load_data(train_path, test_path)

    text_processor = TextProcessor()
    text_processor.build_vocab(train_df['Phrase'].values, min_count=config.MIN_WORD_COUNT)

    train_texts = train_df['Phrase'].values
    train_sentiments = train_df['Sentiment'].values
    test_texts = test_df['Phrase'].values
    test_sentiments = test_df.get('Sentiment', np.zeros(len(test_df)))

    train_texts, val_texts, train_sentiments, val_sentiments = train_test_split(
        train_texts, train_sentiments, test_size=0.2, random_state=42, stratify=train_sentiments
    )

    train_dataset = MovieReviewDataset(train_texts, train_sentiments, text_processor, config.MAX_LEN)
    val_dataset = MovieReviewDataset(val_texts, val_sentiments, text_processor, config.MAX_LEN)
    test_dataset = MovieReviewDataset(test_texts, test_sentiments, text_processor, config.MAX_LEN)

    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False)

    vocab_size = text_processor.vocab_size
    output_size = 5

    model = TextRNN(vocab_size, config.HIDDEN_SIZE, output_size, config.N_LAYER, True).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

    print(f"模型参数: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    train_losses, val_losses, train_accuracies, val_accuracies = [], [], [], []
    start_time = time.time()

    for epoch in range(1, config.N_EPOCHS + 1):
        print(f"\n轮次 {epoch}/{config.N_EPOCHS}")
        print("-" * 50)

        train_loss, train_acc = train_model(model, train_loader, criterion, optimizer, epoch)
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)

        val_loss, val_acc = test_model(model, val_loader, criterion)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)

    training_time = time.time() - start_time
    print(f"训练完成! 总用时: {training_time:.2f}秒")

    print("最终测试结果:")
    test_loss, test_accuracy = test_model(model, test_loader, criterion)

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(range(1, config.N_EPOCHS + 1), train_losses, label='训练损失')
    plt.plot(range(1, config.N_EPOCHS + 1), val_losses, label='验证损失')
    plt.xlabel('轮次')
    plt.ylabel('损失')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(range(1, config.N_EPOCHS + 1), train_accuracies, label='训练准确率')
    plt.plot(range(1, config.N_EPOCHS + 1), val_accuracies, label='验证准确率')
    plt.xlabel('轮次')
    plt.ylabel('准确率 (%)')
    plt.legend()

    plt.tight_layout()
    plt.show()

    model_path = r"D:\潘 +深度学习\movie_sentiment_model.pth"
    torch.save({
        'model_state_dict': model.state_dict(),
        'text_processor': text_processor,
        'config': config
    }, model_path)
    print(f"模型已保存到：{model_path}")


if __name__ == "__main__":
    main()