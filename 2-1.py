import torch

# 定义数据
x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]

# 初始化权重，设置requires_grad=True以跟踪梯度
w = torch.tensor([1.0], requires_grad=True)

# 前向传播函数
def forward(x):
    return x * w

# 损失函数
def loss(x, y):
    y_pred = forward(x)
    return (y_pred - y) ** 2

# 打印训练前的预测结果
print("predict (before training)", 4, forward(4).item())

# 训练过程
for epoch in range(100):
    for x, y in zip(x_data, y_data):
        l = loss(x, y)
        # 反向传播计算梯度
        l.backward()
        print('\tgrad:', x, y, w.grad.item())
        # 更新权重，这里要注意使用w.data来避免计算图的构建
        w.data = w.data - 0.01 * w.grad.data
        # 梯度清零，否则梯度会累加
        w.grad.data.zero_()
    print("progress:", epoch, l.item())

# 打印训练后的预测结果
print("predict (after training)", 4, forward(4).item())