#!/usr/bin/env python
# coding: utf-8

# In[5]:


import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import pennylane as qml
import numpy as np
import torch.nn.functional as F


# 量子设备
n_qubits = 4
dev = qml.device("default.qubit", wires=n_qubits)

# 定义量子电路
@qml.qnode(dev, interface="torch")
def quantum_circuit(inputs, weights):
    for i in range(n_qubits):
        qml.RY(inputs[i], wires=i)  # 用 RY 门将经典输入编码到量子态
    for i in range(n_qubits - 1):
        qml.CNOT(wires=[i, i+1])
    qml.RY(weights[0], wires=0)
    qml.RY(weights[1], wires=1)
    qml.RY(weights[2], wires=2)
    qml.RY(weights[3], wires=3)
    return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

# 定义量子层
class QuantumLayer(nn.Module):
    def __init__(self):
        super(QuantumLayer, self).__init__()
        # 可训练量子参数
        self.weights = nn.Parameter(torch.randn(4))

    def forward(self, x):
        # 只使用前4个特征输入量子电路
        x = x[:, :n_qubits]
        # 将量子电路输出转换为 PyTorch 的 Tensor
        quantum_results = [torch.tensor(quantum_circuit(xi, self.weights),  dtype=torch.float32) for xi in x]
        return torch.stack(quantum_results)

# 定义学生模型（结合经典 CNN 和量子层）
class StudentModel(nn.Module):
    def __init__(self):
        super(StudentModel, self).__init__()
        # 简单的 CNN 层
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, n_qubits)  # 为量子层准备输入
        self.quantum_layer = QuantumLayer()
        self.fc3 = nn.Linear(n_qubits, 100)  # CIFAR-100 有100个类别

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 32 * 8 * 8)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)  # 特征映射到量子层输入
        x = self.quantum_layer(x)
        x = self.fc3(x)  # 最终分类
        return x

# 定义知识蒸馏损失
def distillation_loss(student_output, teacher_output, temperature=3.0):
    soft_teacher_output = nn.functional.softmax(teacher_output / temperature, dim=1)
    soft_student_output = nn.functional.log_softmax(student_output / temperature, dim=1)
    return nn.KLDivLoss(reduction='batchmean')(soft_student_output, soft_teacher_output)

# 数据预处理
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# CIFAR-100 数据集加载
trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=False, num_workers=2)

# 初始化教师模型 (这里可以使用预训练的 ResNet 或其他模型)
teacher_model = torchvision.models.resnet18(pretrained=True)
teacher_model.fc = nn.Linear(512, 100)  # 修改输出层以适应 CIFAR-100
teacher_model.eval()  # 设置为评估模式

# 初始化学生模型
student_model = StudentModel()
criterion = nn.CrossEntropyLoss()  # 分类损失
optimizer = optim.Adam(student_model.parameters(), lr=0.001)

# 训练学生模型
for epoch in range(10):  # 训练10个epoch
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data

        # 获取教师模型的输出 (soft target)
        with torch.no_grad():
            teacher_output = teacher_model(inputs)

        # 学生模型前向传播
        optimizer.zero_grad()
        student_output = student_model(inputs)

        # 计算知识蒸馏损失
        loss = distillation_loss(student_output, teacher_output) + criterion(student_output, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % 100 == 99:    # 每100个小批量输出一次损失
            print(f'[Epoch {epoch + 1}, Batch {i + 1}] loss: {running_loss / 100:.3f}')
            running_loss = 0.0

print('Finished Training')

# 测试学生模型
correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = student_model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy of the student model on the CIFAR-100 test images: {100 * correct / total}%')

