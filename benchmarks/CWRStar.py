#!/usr/bin/env python
# coding: utf-8

# In[2]:


import torch
import torch.nn as nn
import torch.optim as optim

from avalanche.benchmarks import SplitMNIST
from avalanche.logging import InteractiveLogger
from avalanche.training import CWRStar

# --------------------
# 1. 定义经典 CNN 模型
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
# 2. 构建 SplitMNIST 基准
# --------------------
benchmark = SplitMNIST(
    n_experiences=5,
    return_task_id=False
)

# --------------------
# 3. 实例化模型、优化器与损失
# --------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = SimpleCNN(num_classes=10).to(device)
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
criterion = nn.CrossEntropyLoss()

# --------------------
# 4. 配置 CWR* 策略
# --------------------
# 使用 CWRStar 来实现 class-incremental learning，
# 会为最后一层 fc2 维护 task-specific 权重
strategy = CWRStar(
    model=model,
    optimizer=optimizer,
    criterion=criterion,
    cwr_layer_name=None,    # None 表示自动使用模型最后一层
    train_mb_size=32,
    train_epochs=20,
    eval_mb_size=64,
    device=device
)

# 日志设置
logger = InteractiveLogger()
strategy.evaluator.loggers = [logger]

# --------------------
# 5. 持续训练与评估
# --------------------
task_accuracies = []
print("Starting SplitMNIST + CWRStar...\n")
for experience in benchmark.train_stream:
    print(f"--- Training experience {experience.current_experience} ---")
    strategy.train(experience)
    print("Evaluation on all experiences:")
    task_accuracies.append(strategy.eval(benchmark.test_stream))
    

print("Training completed!")


# In[ ]:


import os
# Define the file path
file_path = "/home/yangz2/code/quantum_cl/results/list/splitminist_CWRStar.pkl"

# Create directories if they don't exist
os.makedirs(os.path.dirname(file_path), exist_ok=True)  # <-- Add this line   


# In[ ]:


import pickle
# 存储到文件
with open(file_path, "wb") as f:
    pickle.dump([task_accuracies], f)  

