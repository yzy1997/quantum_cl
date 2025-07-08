#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch
import torch.nn as nn
import torch.optim as optim

from avalanche.benchmarks import SplitMNIST
from avalanche.logging import InteractiveLogger, TextLogger, TensorboardLogger
from avalanche.training import Naive
from avalanche.training.plugins.ewc import EWCPlugin

# --------------------
# 1. 定义简单 CNN 模型
# --------------------
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5, padding=2)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, padding=2)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2)

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64 * 7 * 7, 256)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.flatten(x)
        x = self.relu3(self.fc1(x))
        return self.fc2(x)

# --------------------
# 2. 创建 SplitMNIST 基准
# --------------------
benchmark = SplitMNIST(
    n_experiences=5,
    return_task_id=False
)

# --------------------
# 3. 设置模型、优化器、损失和 EWC 插件
# --------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = SimpleCNN(num_classes=10).to(device)
optimizer = optim.SGD(model.parameters(), lr=0.005, momentum=0.9)
criterion = nn.CrossEntropyLoss()

# Elastic Weight Consolidation 插件
ewc_plugin = EWCPlugin(
    ewc_lambda=200,
    mode='online',
    decay_factor=0.5
)

# --------------------
# 4. 定义 Naive 策略并添加 EWC
# --------------------
strategy = Naive(
    model=model,
    optimizer=optimizer,
    criterion=criterion,
    train_mb_size=128,
    train_epochs=60,
    eval_mb_size=258,
    device=device,
    plugins=[ewc_plugin]
)

# 日志记录
interactive_logger = InteractiveLogger()
# text_logger = TextLogger(open('ewc_splitmnist.log', 'w'))
# tb_logger = TensorboardLogger('ewc_tb_logs')
strategy.evaluator.loggers = [interactive_logger]

# --------------------
# 5. 训练与评估
# --------------------
task_accuracies = []
print("Starting SplitMNIST + EWC...")
for experience in benchmark.train_stream:
    print(f"\n--- Training experience {experience.current_experience} ---")
    strategy.train(experience)
    print("Evaluation:")
    resutls = strategy.eval(benchmark.test_stream)
    task_accuracies.append(resutls)

print("Training completed!")


# In[ ]:


import os
# Define the file path
file_path = "/home/yangz2/code/quantum_cl/results/list/splitminist_EWC.pkl"

# Create directories if they don't exist
os.makedirs(os.path.dirname(file_path), exist_ok=True)  # <-- Add this line   


# In[ ]:


import pickle
# 存储到文件
with open(file_path, "wb") as f:
    pickle.dump([task_accuracies], f)  

