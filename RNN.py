import torch

idx2char = ['d', 'l', 'e', 'a', 'r', 'n']
char2idx = {char: idx for idx, char in enumerate(idx2char)}
input_str = "dlearn"
target_str = "lanrla"

x_data = [char2idx[char] for char in input_str]
y_data = [char2idx[char] for char in target_str]

input_size = len(idx2char)
hidden_size = 6
batch_size = 1
seq_len = len(input_str)
num_layers = 1

one_hot_lookup = torch.eye(input_size)
x_one_hot = one_hot_lookup[x_data]
inputs = x_one_hot.view(seq_len, batch_size, input_size)
labels = torch.LongTensor(y_data)


class Model(torch.nn.Module):
    def __init__(self, input_size, hidden_size, batch_size, num_layers=1):
        super(Model, self).__init__()
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.rnn = torch.nn.RNN(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=False
        )

    def forward(self, input):
        hidden = torch.zeros(self.num_layers, self.batch_size, self.hidden_size)
        out, _ = self.rnn(input, hidden)
        return out.view(-1, self.hidden_size)


net = Model(input_size, hidden_size, batch_size, num_layers)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=0.05)

train_log = []
for epoch in range(30):
    optimizer.zero_grad()
    outputs = net(inputs)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()

    _, idx = outputs.max(dim=1)
    predicted_str = ''.join([idx2char[x.item()] for x in idx])

    epoch_loss = loss.item()
    train_log.append({
        'epoch': epoch + 1,
        'loss': epoch_loss,
        'predicted': predicted_str,
        'target': target_str
    })

    print('Epoch [%d/30], Loss: %.4f, Predicted: %s, Target: %s' %
          (epoch + 1, epoch_loss, predicted_str, target_str))

print("\nRNN训练完成！")
print('最终预测结果:', train_log[-1]["predicted"])
print('目标结果:', target_str)

import pandas as pd

pd.DataFrame(train_log).to_csv('rnn_train_log.csv', index=False)
print("训练日志已保存到 rnn_train_log.csv")