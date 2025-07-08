#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from avalanche.benchmarks import SplitMNIST
from avalanche.logging import InteractiveLogger, TextLogger, TensorboardLogger
from avalanche.evaluation.metrics import accuracy_metrics, loss_metrics, forgetting_metrics
from avalanche.training import Naive
from avalanche.training.plugins import SynapticIntelligencePlugin

# --------------------
# 1. 网络定义
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
        x = self.fc2(x)
        return x


# --------------------
# 2. 基准 (Benchmark) 构建
# --------------------
# 使用 SplitMNIST: 5 个小任务, 每个任务包含 2 个类别 (0-1, 2-3, ...)
# 使用 SplitMNIST: 5 个小任务，随机类别顺序
benchmark = SplitMNIST(
    n_experiences=5,
    return_task_id=False
)

# --------------------
# 3. 模型、优化器、损失和插件
# --------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = SimpleCNN(num_classes=10).to(device)
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
criterion = nn.CrossEntropyLoss()

# Synaptic Intelligence 插件
si_plugin = SynapticIntelligencePlugin(
    si_lambda=0.1,   # 正则化强度, 可根据实验调整
    eps=0.5          # 防止除零
)

# --------------------
# 4. 定义训练策略
# --------------------
strategy = Naive(
    model=model,
    optimizer=optimizer,
    criterion=criterion,
    train_mb_size=32,
train_epochs=40,
    eval_mb_size=64,
    device=device,
    plugins=[si_plugin]
)

# 日志记录
interactive_logger = InteractiveLogger()
# text_logger = TextLogger(open('si_splitmnist_logs.txt', 'w'))
# tensorboard_logger = TensorboardLogger('tb_logs')
strategy.evaluator.loggers = [interactive_logger]

# --------------------
# 5. 训练与评估
# --------------------
task_accuracies = []
print("Starting continual training on SplitMNIST with SI...")
for experience in benchmark.train_stream:
    print(f"\n--- Training on experience {experience.current_experience} ---")
    strategy.train(experience)
    print("Evaluation:")
    results = strategy.eval(benchmark.test_stream)
    task_accuracies.append(results)

print("Training completed!")


# In[ ]:


import os
# Define the file path
file_path = "/home/yangz2/code/quantum_cl/results/list/splitminist_SI.pkl"

# Create directories if they don't exist
os.makedirs(os.path.dirname(file_path), exist_ok=True)  # <-- Add this line   


# In[ ]:


import pickle
# 存储到文件
with open(file_path, "wb") as f:
    pickle.dump([task_accuracies], f)  

