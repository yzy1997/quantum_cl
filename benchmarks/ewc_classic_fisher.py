#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#!/usr/bin/env python
# coding: utf-8

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms
from sklearn.decomposition import IncrementalPCA
from avalanche.benchmarks.classic import SplitMNIST
from avalanche.training import EWC
from avalanche.training.plugins import EvaluationPlugin
from avalanche.evaluation.metrics import forgetting_metrics, \
accuracy_metrics, loss_metrics, timing_metrics, cpu_usage_metrics, \
confusion_matrix_metrics, disk_usage_metrics
from avalanche.logging import InteractiveLogger
from avalanche.benchmarks import nc_benchmark
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os

# 设备设置
device = "cuda:1" if torch.cuda.is_available() else "cpu"

# 普通CNN网络定义
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        # 输入通道为1（灰度图像），输出通道为16，卷积核大小3x3
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(32 * 7 * 7, 128)  # 假设输入是28x28，经过两次池化后为7x7
        self.fc2 = nn.Linear(128, num_classes)
        
    def forward(self, x):
        # 输入形状: [batch, 256] -> 需要reshape为图像格式
        # 假设原始图像是16x16 (256=16x16)，但我们知道MNIST是28x28
        # 这里我们reshape为28x28 (784个像素)，但我们只有256个特征
        # 所以我们需要上采样或填充 - 这里我们使用简单的线性层进行转换
        
        # 如果输入是256维特征，先转换为784维
        if x.shape[1] == 256:
            x = x.view(-1, 1, 16, 16)  # 16x16=256
            # 上采样到28x28
            x = F.interpolate(x, size=(28, 28), mode='bilinear', align_corners=False)
        else:
            # 如果输入已经是图像格式
            x = x.view(-1, 1, 28, 28)
        
        # 卷积层
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)  # 28x28 -> 14x14
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)  # 14x14 -> 7x7
        
        # 展平
        x = x.view(-1, 32 * 7 * 7)
        
        # 全连接层
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Step 1: 加载预处理好的 PCA 数据
with open("/home/yangz2/code/quantum_cl/data/splitmnist_pca256.pkl", 'rb') as f:
    processed = pickle.load(f)

# Step 2: 重构每个经验的 TensorDatasets（train/val）
datasets_by_exp = {}
for vec, label, exp_id in processed:
    datasets_by_exp.setdefault(exp_id, {"X": [], "y": []})
    datasets_by_exp[exp_id]["X"].append(vec)
    datasets_by_exp[exp_id]["y"].append(label)

# Step 3: 将每个 experience 的数据打包为 TensorDatasets
train_datasets = []
test_datasets = []
task_labels = []
for exp_id in sorted(datasets_by_exp):
    X = torch.stack([torch.tensor(v) for v in datasets_by_exp[exp_id]["X"]])
    y = torch.tensor(datasets_by_exp[exp_id]["y"], dtype=torch.long)
    
    ds = TensorDataset(X, y)
    
    train_datasets.append(ds)
    test_datasets.append(ds)
    task_labels.append(0)

# Step 4: 使用 nc_benchmark 创建持续学习基准
benchmark = nc_benchmark(
    train_datasets, 
    test_datasets, 
    n_experiences=len(train_datasets),
    task_labels=task_labels
)

print("✔ 使用预处理好的 PCA 数据创建了 nc_benchmark")

# -----------------------------------------------------------------------------
# 设置训练环境
# -----------------------------------------------------------------------------
model = SimpleCNN(num_classes=10).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# Evaluation setup
interactive_logger = InteractiveLogger()
eval_plugin = EvaluationPlugin(
    accuracy_metrics(minibatch=True, epoch=True, experience=True, stream=True),
    loss_metrics(minibatch=True, epoch=True, experience=True, stream=True),
    timing_metrics(epoch=True, epoch_running=True),
    cpu_usage_metrics(experience=True),
    forgetting_metrics(experience=True, stream=True),
    confusion_matrix_metrics(num_classes=10, save_image=False, stream=True),
    disk_usage_metrics(minibatch=True, epoch=True, experience=True, stream=True),
    loggers=[interactive_logger]
)


# -----------------------------------------------------------------------------
# 使用带有Fisher Matrix的EWC持续学习策略
# -----------------------------------------------------------------------------
strategy = EWC(
    model=model,
    optimizer=optimizer,
    criterion=criterion,
    ewc_lambda=0.4,        # 调整正则化强度
    mode='separate',       # 使用独立Fisher矩阵
    decay_factor=None,     # 禁用衰减
    train_epochs=10,
    device=device,
    evaluator=eval_plugin
)

# -----------------------------------------------------------------------------
# 开始训练与评估
# -----------------------------------------------------------------------------
task_accuracies = []
save_dir = "/home/yangz2/code/quantum_cl/results/list"
os.makedirs(save_dir, exist_ok=True)

print("Starting training...")
for experience in benchmark.train_stream:
    print(f"\n--- Training on experience {experience.current_experience} ---")
    strategy.train(experience)
    
    print(f"--- Evaluating after experience {experience.current_experience} ---")
    results = strategy.eval(benchmark.test_stream)
    task_accuracies.append(results)
    
    # 保存中间结果
    with open(os.path.join(save_dir, f"splitmnist_EWC_classic_cnn_interim_results_exp_{experience.current_experience}.pkl"), "wb") as f:
        pickle.dump(task_accuracies, f)

# 保存最终结果
with open(os.path.join(save_dir, "splitmnist_EWC_classic_cnn_final.pkl"), "wb") as f:
    pickle.dump(task_accuracies, f)

print("✔ Training and evaluation completed!")

