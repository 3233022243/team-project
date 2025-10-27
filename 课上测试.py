'''
1. 准备工具→2. 做数据盒子→3. 处理原始数据→4. 拆分训练 / 测试集→5. 搭预测模型
→6. 设定训练规则→7. 反复训练并记录效果→8. 画图看结果→9. 用最好的模型算最终成绩。
'''
# 导入必要的库
import torch  # PyTorch深度学习框架
import torch.nn as nn  # 神经网络模块
import torch.optim as optim  # 优化器模块
from torch.utils.data import Dataset, DataLoader  # 数据集和数据加载器
import pandas as pd  # 数据处理库
import numpy as np  # 数值计算库
import matplotlib.pyplot as plt  # 可视化库

# 设置Matplotlib字体，解决中文显示问题
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示异常

from sklearn.preprocessing import LabelEncoder, StandardScaler  # 数据预处理工具
from sklearn.model_selection import train_test_split  # 数据集划分工具
from sklearn.metrics import accuracy_score  # 分类评估指标
import os  # 操作系统交互

# 设置随机种子，确保实验可复现（每次运行结果一致）
torch.manual_seed(42)  # PyTorch随机种子
np.random.seed(42)  # NumPy随机种子


# --------------------------
# 1. 定义数据集类（继承PyTorch的Dataset）
# 作用：将特征和标签封装为PyTorch可识别的格式，方便后续加载
# --------------------------
class StudentDataset(Dataset):
    def __init__(self, features, labels):
        # 将特征和标签转换为PyTorch张量（Tensor）
        # 特征用float32类型（神经网络输入通常为32位浮点数）
        self.features = torch.tensor(features, dtype=torch.float32)
        # 标签用long类型（多分类任务的标签需为长整型）
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        # 返回数据集的总样本数（必须实现的方法）
        return len(self.features)

    def __getitem__(self, idx):
        # 根据索引返回单条数据（特征+标签）（必须实现的方法）
        return self.features[idx], self.labels[idx]


# --------------------------
# 2. 数据加载与预处理函数
# 作用：读取原始数据，处理缺失值，转换特征和标签格式，为模型输入做准备
# --------------------------
def load_and_preprocess_data(file_path):
    # 读取CSV格式的数据集
    df = pd.read_csv(file_path)

    # 数据清洗：检查并处理缺失值
    print(f"原始数据形状: {df.shape}")  # 输出原始数据的行数和列数
    print(f"各列缺失值数量:\n{df.isnull().sum()}\n")  # 统计每列的缺失值

    # 处理数值型特征的缺失值：用中位数填充（中位数比均值更抗极端值）
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns  # 筛选数值列
    for col in numeric_cols:
        if df[col].isnull().sum() > 0:  # 若该列有缺失值
            df[col].fillna(df[col].median(), inplace=True)  # 用中位数填充
            print(f"数值列 {col} 缺失值已用中位数填充")

    # 处理分类型特征的缺失值：用众数填充（众数是出现频率最高的值）
    categorical_cols = df.select_dtypes(include=['object']).columns  # 筛选字符串列
    for col in categorical_cols:
        if df[col].isnull().sum() > 0:  # 若该列有缺失值
            df[col].fillna(df[col].mode()[0], inplace=True)  # 用众数填充
            print(f"分类型列 {col} 缺失值已用众数填充")

    # 最终清理：删除仍有缺失值的行（极端情况处理）
    df.dropna(inplace=True)
    print(f"清洗后数据形状: {df.shape}\n")  # 输出清洗后的数据形状

    # 分离特征（X）和标签（y）
    # 假设目标列是'G3'（学生最终成绩），特征为其他所有列
    X = df.drop('G3', axis=1)  # 特征：排除目标列
    y = df['G3']  # 标签：目标列（最终成绩）

    # 将连续的成绩转换为多分类标签
    # 学生成绩通常为0-20分，这里按区间分为5个等级（0-4）
    y = pd.cut(y, bins=5, labels=[0, 1, 2, 3, 4]).astype(int)

    # 处理分类型特征：将字符串转换为数值（神经网络只能处理数值输入）
    for col in categorical_cols:
        if col != 'G3':  # 排除标签列（如果标签列是字符串的话）
            le = LabelEncoder()  # 初始化标签编码器
            X[col] = le.fit_transform(X[col])  # 将字符串映射为0,1,2...

    # 特征标准化：将所有特征缩放到均值为0、方差为1的范围
    # 目的：消除不同特征的量纲影响，使模型训练更稳定
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)  # 标准化处理

    return X_scaled, y.values  # 返回处理后的特征和标签


# --------------------------
# 3. 主函数（程序入口）
# 作用：串联所有步骤，完成从数据加载到模型评估的全流程
# --------------------------
def main():
    # 加载数据（替换为你的数据文件路径）
    file_path = "C:\\Users\\dell\\Desktop\\student-por.csv"
    X, y = load_and_preprocess_data(file_path)  # 得到预处理后的特征和标签

    # 划分训练集和测试集
    # test_size=0.2：80%数据用于训练，20%用于测试
    # random_state=42：固定随机划分方式，保证可复现
    # stratify=y：按标签分布比例划分，避免某类样本在测试集中过多/过少
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 将训练集和测试集封装为自定义的StudentDataset
    train_dataset = StudentDataset(X_train, y_train)
    test_dataset = StudentDataset(X_test, y_test)

    # 定义数据加载器（DataLoader）
    # 作用：按批次加载数据，支持多进程加速（此处因Windows系统禁用）
    batch_size = 32  # 每次加载32个样本（批次大小）
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,  # 训练集打乱顺序，避免模型学习数据顺序
        num_workers=0  # Windows系统设置为0，防止多进程冲突
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,  # 测试集不打乱，保证结果可复现
        num_workers=0
    )

#--------------较难：代码中的input_dim：输入维度，就是学生的特征数（30 个），相当于预测器的 “入口宽度”；
#--------------num_classes：输出类别数，就是成绩等级数（5 个），相当于预测器的 “出口宽度”。

    # --------------------------
    # 4. 定义神经网络模型
    # 作用：构建用于分类的神经网络结构
    # --------------------------
    class StudentGradeModel(nn.Module):
        def __init__(self, input_dim, num_classes):
            super(StudentGradeModel, self).__init__()  # 继承父类初始化
            # 定义网络层（使用Sequential按顺序组合层）
            self.layers = nn.Sequential(
                # 第一层：输入层→隐藏层1
                nn.Linear(input_dim, 64),  # 线性层：输入维度→64维
                nn.ReLU(),  # 激活函数：引入非线性，解决线性不可分问题
                nn.BatchNorm1d(64),  # 批归一化：使每层输入分布稳定，加速训练
                nn.Dropout(0.2),  # 随机丢弃20%神经元：防止过拟合

                # 第二层：隐藏层1→隐藏层2
                nn.Linear(64, 32),  # 线性层：64维→32维
                nn.ReLU(),
                nn.BatchNorm1d(32),
                nn.Dropout(0.1),  # 随机丢弃10%神经元

                # 第三层：输出层
                nn.Linear(32, num_classes)  # 线性层：32维→输出类别数（5类）
            )

        def forward(self, x):
            # 前向传播：定义数据在网络中的流动路径
            return self.layers(x)  # 输入x经过所有层的处理后输出

    # 初始化模型
    input_dim = X_train.shape[1]  # 输入维度=特征数（30个）
    num_classes = len(np.unique(y))  # 输出类别数=成绩等级数（5类）
    model = StudentGradeModel(input_dim, num_classes)  # 创建模型实例

    # --------------------------
    # 5. 训练配置
    # 作用：设置损失函数、优化器等训练相关参数
    # --------------------------
    criterion = nn.CrossEntropyLoss()  # 多分类任务常用的损失函数（内置SoftMax）
    # 优化器：Adam（自适应学习率优化器，训练更稳定）
    # lr=0.0005：学习率（步长），weight_decay=1e-5：L2正则化（防止过拟合）
    optimizer = optim.Adam(model.parameters(), lr=0.0005, weight_decay=1e-5)
    # 自动选择计算设备（有GPU用GPU，否则用CPU）
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)  # 将模型移动到计算设备上

    # 梯度裁剪阈值：防止梯度爆炸（梯度太大导致参数更新异常）
    clip_value = 1.0

    # 训练参数
    num_epochs = 50  # 训练总轮次（整个数据集训练50遍）
    best_accuracy = 0.0  # 记录最佳测试准确率
    best_model_path = 'best_student_model.pt'  # 最佳模型保存路径

    # 记录训练过程中的指标（用于后续可视化）
    train_losses = []  # 训练损失
    train_accuracies = []  # 训练准确率
    test_losses = []  # 测试损失
    test_accuracies = []  # 测试准确率

    # --------------------------
    # 6. 模型训练循环
    # 作用：通过多轮迭代，让模型学习特征与标签的关系
    # --------------------------
    for epoch in range(num_epochs):
        # 训练阶段：开启模型训练模式（启用Dropout等）
        model.train()
        train_running_loss = 0.0  # 累计训练损失
        train_preds = []  # 存储训练集预测结果
        train_targets = []  # 存储训练集真实标签

        # 遍历训练集的每个批次
        for features, labels in train_loader:
            # 将数据移动到计算设备（GPU/CPU）
            features, labels = features.to(device), labels.to(device)

            # 前向传播：用当前模型预测结果
            outputs = model(features)
            # 计算损失（预测结果与真实标签的差距）
            loss = criterion(outputs, labels)

            # 反向传播与参数更新
            optimizer.zero_grad()  # 清零梯度（防止上一轮梯度累积）
            loss.backward()  # 反向传播：计算梯度
            nn.utils.clip_grad_norm_(model.parameters(), clip_value)  # 梯度裁剪
            optimizer.step()  # 更新模型参数

            # 累计损失（乘以批次大小，最后求平均）
            train_running_loss += loss.item() * features.size(0)
            # 取预测概率最大的类别作为预测结果
            _, preds = torch.max(outputs, 1)
            # 保存预测结果和真实标签（需移回CPU才能转换为NumPy）
            train_preds.extend(preds.cpu().numpy())
            train_targets.extend(labels.cpu().numpy())

        # 计算本轮训练的平均损失和准确率
        train_epoch_loss = train_running_loss / len(train_dataset)
        train_epoch_acc = accuracy_score(train_targets, train_preds)
        # 记录指标
        train_losses.append(train_epoch_loss)
        train_accuracies.append(train_epoch_acc)

        # 测试阶段：关闭模型训练模式（禁用Dropout等）
        model.eval()
        test_running_loss = 0.0  # 累计测试损失
        test_preds = []  # 存储测试集预测结果
        test_targets = []  # 存储测试集真实标签

        # 测试时不计算梯度（节省内存，加速计算）
        with torch.no_grad():
            # 遍历测试集的每个批次
            for features, labels in test_loader:
                features, labels = features.to(device), labels.to(device)
                outputs = model(features)  # 前向传播预测
                loss = criterion(outputs, labels)  # 计算测试损失

                # 累计测试损失
                test_running_loss += loss.item() * features.size(0)
                # 取预测结果
                _, preds = torch.max(outputs, 1)
                test_preds.extend(preds.cpu().numpy())
                test_targets.extend(labels.cpu().numpy())

        # 计算本轮测试的平均损失和准确率
        test_epoch_loss = test_running_loss / len(test_dataset)
        test_epoch_acc = accuracy_score(test_targets, test_preds)
        # 记录指标
        test_losses.append(test_epoch_loss)
        test_accuracies.append(test_epoch_acc)

        # 保存性能最佳的模型（只保存测试准确率最高的模型）
        if test_epoch_acc > best_accuracy:
            best_accuracy = test_epoch_acc
            torch.save(model.state_dict(), best_model_path)  # 保存模型参数
            print(f'Epoch {epoch + 1}: 最佳模型已保存 (准确率: {best_accuracy:.4f})')

        # 打印本轮训练和测试的指标
        print(f'Epoch [{epoch + 1}/{num_epochs}]')
        print(f'训练损失: {train_epoch_loss:.4f}, 训练准确率: {train_epoch_acc:.4f}')
        print(f'测试损失: {test_epoch_loss:.4f}, 测试准确率: {test_epoch_acc:.4f}\n')

    # --------------------------
    # 7. 训练过程可视化
    # 作用：通过图表直观展示模型训练的收敛情况
    # --------------------------
    plt.figure(figsize=(12, 5))  # 创建画布（宽12，高5）

    # 第一个子图：损失曲线
    plt.subplot(1, 2, 1)  # 1行2列，第1个图
    plt.plot(range(1, num_epochs + 1), train_losses, label='训练损失')
    plt.plot(range(1, num_epochs + 1), test_losses, label='测试损失')
    plt.xlabel('Epoch（轮次）')
    plt.ylabel('损失值')
    plt.title('训练与测试损失曲线')
    plt.legend()  # 显示图例

    # 第二个子图：准确率曲线
    plt.subplot(1, 2, 2)  # 1行2列，第2个图
    plt.plot(range(1, num_epochs + 1), train_accuracies, label='训练准确率')
    plt.plot(range(1, num_epochs + 1), test_accuracies, label='测试准确率')
    plt.xlabel('Epoch（轮次）')
    plt.ylabel('准确率')
    plt.title('训练与测试准确率曲线')
    plt.legend()

    plt.tight_layout()  # 自动调整子图间距
    plt.show()  # 显示图表

    # --------------------------
    # 8. 加载最佳模型并评估
    # 作用：验证最佳模型的最终性能
    # --------------------------
    # 创建新的模型实例并加载最佳参数
    best_model = StudentGradeModel(input_dim, num_classes)
    best_model.load_state_dict(torch.load(best_model_path))  # 加载保存的参数
    best_model.to(device)  # 移动到计算设备
    best_model.eval()  # 开启评估模式

    # 用最佳模型在测试集上重新预测
    test_preds = []
    test_targets = []
    with torch.no_grad():  # 不计算梯度
        for features, labels in test_loader:
            features, labels = features.to(device), labels.to(device)
            outputs = best_model(features)
            _, preds = torch.max(outputs, 1)
            test_preds.extend(preds.cpu().numpy())
            test_targets.extend(labels.cpu().numpy())

    # 计算并输出最终准确率
    final_accuracy = accuracy_score(test_targets, test_preds)
    print(f'最佳模型在测试集上的准确率: {final_accuracy:.4f}')


# 程序入口：当脚本直接运行时执行main()函数
# 作用：符合Python编程规范，避免被导入时自动执行
if __name__ == '__main__':
    main()