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

one_hot_lookup = torch.eye(input_size)
x_one_hot = one_hot_lookup[x_data]
inputs = x_one_hot.view(seq_len, batch_size, input_size)
labels = torch.LongTensor(y_data).view(-1, 1)


class Model(torch.nn.Module):
    def __init__(self, input_size, hidden_size, batch_size):
        super(Model, self).__init__()
        self.batch_size = batch_size
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.rnncell = torch.nn.RNNCell(input_size=input_size, hidden_size=hidden_size)

    def forward(self, input, hidden):
        hidden = self.rnncell(input, hidden)
        return hidden

    def init_hidden(self):
        return torch.zeros(self.batch_size, self.hidden_size)


net = Model(input_size, hidden_size, batch_size)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=0.1)

train_log = []
for epoch in range(50):
    loss = 0.0
    optimizer.zero_grad()
    hidden = net.init_hidden()
    predicted_str = []

    for input, label in zip(inputs, labels):
        hidden = net(input, hidden)
        loss += criterion(hidden, label)

        _, idx = hidden.max(dim=1)
        predicted_str.append(idx2char[idx.item()])

    loss.backward()
    optimizer.step()

    epoch_loss = loss.item()
    pred_str = ''.join(predicted_str)
    train_log.append({
        'epoch': epoch + 1,
        'loss': epoch_loss,
        'predicted': pred_str,
        'target': target_str
    })

    print('Epoch [%d/50], Loss: %.4f, Predicted: %s, Target: %s' %
          (epoch + 1, epoch_loss, pred_str, target_str))

print("\nRNNCell训练完成！")
print('最终预测结果:', train_log[-1]["predicted"])
print('目标结果:', target_str)