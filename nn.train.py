import torch.nn as nn
import torch.optim
import torchvision
from torch.utils.tensorboard import SummaryWriter

from model.nn_model import *
from torch.utils.data import DataLoader
import time

# 训练设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 数据归一化
transform = torchvision.transforms.Compose(
    [torchvision.transforms.ToTensor(),
     torchvision.transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))]
)
# 准备数据集
train_data = torchvision.datasets.CIFAR10(root="./dataset",train=True,transform=transform,download=True)
test_data = torchvision.datasets.CIFAR10(root="./dataset",train=False,transform=transform,download=True)

train_data_size = len(train_data)
test_data_size = len(test_data)
print("训练集的长度为：{}".format(train_data_size))
print("训练集的长度为：{}".format(test_data_size))

# 利用dataloader进行数据加载
train_dataloader = DataLoader(train_data,batch_size=64,shuffle=True)
test_dataloader = DataLoader(test_data,batch_size=64,shuffle=False)

# 创建网络模型
mynn = Mynn()
mynn.to(device)

# 损失函数
loss_fn = nn.CrossEntropyLoss()

# 优化器
optimizer = torch.optim.SGD(mynn.parameters(), lr=0.01)

# 设置训练网络的参数
# 记录训练次数
total_train_step = 0
total_test_step = 0
# 要训练次数
epoch = 35

#添加tensorboard
write = SummaryWriter("../log_train_raise")

start_time = time.time()

with open('./result/result.txt','w')as f:
    for i in range(epoch):
        print(f"-----第{i+1}次训练开始-----".format(i+1))
        f.write(f"-----第{i+1}次训练开始-----".format(i+1)+'\n')
        # 训练步骤开始
        total_train_loss = 0
        total_train_accuracy = 0
        for data in train_dataloader:
            img, targets = data
            img = img.to(device)
            targets = targets.to(device)
            outputs = mynn(img)
            loss = loss_fn(outputs, targets)
            total_train_loss = total_train_loss+loss.item()
            train_accuracy = (outputs.argmax(1) == targets).sum()
            total_train_accuracy = total_train_accuracy+train_accuracy
            # 优化器优化模型
            optimizer.zero_grad()  # 梯度清零
            loss.backward()
            optimizer.step()
            total_train_step = total_train_step + 1
            if total_train_step % 100==0:
                print("提示：训练集第{}次训练,loss:{}".format(total_train_step,loss.item()))
                f.write("提示：训练集第{}次训练,loss:{}".format(total_train_step,loss.item())+'\n')
                write.add_scalar("train__loss_raise_1",loss.item(),total_train_step)
        print('第{}次训练集上的总loss:{}'.format(i+1, total_train_loss))
        f.write('第{}次训练集上的总loss:{}'.format(i+1, total_train_loss) + '\n')
        print('第{}次训练集上的准确率:{}'.format(i+1, total_train_accuracy/train_data_size))
        f.write('第{}次训练集上的准确率:{}'.format(i+1, total_train_accuracy/train_data_size) + '\n')

        # 测速步骤开始
        total_test_loss = 0
        total_test_accuracy = 0
        with torch.no_grad():
            for data in test_dataloader:
                img, targets = data
                img = img.to(device)
                targets = targets.to(device)
                outputs = mynn(img)
                loss = loss_fn(outputs,targets)
                total_test_loss = total_test_loss+loss.item()
                test_accuracy = (outputs.argmax(1) == targets).sum()
                total_test_accuracy = total_test_accuracy+test_accuracy
            print("第{}次整体体测试集上的总loss:{}".format(i+1,total_test_loss))
            f.write("第{}次整体体测试集上的总loss:{}".format(i+1,total_test_loss) + '\n')
            print("第{}次整体测试集上的正确率:{}".format(i+1,total_test_accuracy / test_data_size))
            f.write("第{}次整体测试集上的正确率:{}".format(i+1,total_test_accuracy / test_data_size) + '\n')
            write.add_scalar("test__loss_raise_1",total_test_loss,total_test_step)
            write.add_scalar("test__accuracy_raise_1",total_test_accuracy/test_data_size,total_test_step)
            total_test_step = total_test_step + 1

        # 保存模型
        # 加了state_dict是官方保存模式，保存为字典型
    torch.save(mynn.state_dict(),'./{}.pth'.format(time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())))
    torch.save(mynn,'./{}.pkl'.format(time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())))
    write.close()
    end_time = time.time()
    use_time = end_time - start_time
    print("一共用时{}分钟".format(use_time/60))
    f.write("一共用时{}分钟".format(use_time/60)+'\n')
    f.close()