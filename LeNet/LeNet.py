import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt


# 定义LeNet-5网络结构
class LeNet5(nn.Module):
    # 初始化实例
    def __init__(self):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    # 定义向前传播，描述流动路径
    def forward(self, x):
        x = self.pool1(torch.relu(self.conv1(x)))
        x = self.pool2(torch.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# 数据预处理，包括随机水平翻转、随机裁剪、转换为张量、归一化操作
transform_train = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# 数据预处理，包括转换为张量、归一化操作
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# 加载数据集
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# 测试准确率的函数，遍历测试集测试准确率
def test(net):
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    # print(f"Test Accuracy: {accuracy}%")

    return accuracy

# 训练函数
def train(net, criterion, optimizer, num_epochs):

    # 记录loss和accuracy
    train_loss = []
    train_accuracy = []
    # 进行num_epochs次
    for epoch in range(num_epochs):
        running_loss = 0.0
        # 遍历训练数据加载器中的批次
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            # 将优化器的梯度置零
            optimizer.zero_grad()

            outputs = net(inputs)
            loss = criterion(outputs, labels)
            # 进行反向传播，计算梯度
            loss.backward()
            # 更新模型
            optimizer.step()

            running_loss += loss.item()
            if i % 200 == 199:
                train_loss.append(running_loss / 200)
                running_loss = 0.0

        accuracy = test(net)
        train_accuracy.append(accuracy)
        print(f"Epoch {epoch + 1}/{num_epochs} - Loss: {train_loss[-1]} - Accuracy: {accuracy}")

    return train_loss, train_accuracy


# 定义测试函数



def main():
    # 创建LeNet-5网络模型实例
    net = LeNet5()

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    num_epochs = 50
    train_loss, train_accuracy = train(net, criterion, optimizer, num_epochs)
    test(net)

    # 可视化训练过程
    plt.plot(train_loss)
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.savefig("loss.jpg")

    plt.clf()

    plt.plot(train_accuracy)
    plt.xlabel('Iterations')
    plt.ylabel('Accuracy')
    plt.title('Training Accuracy')
    plt.savefig("accuracy.jpg")


if __name__ == '__main__':
    main()
