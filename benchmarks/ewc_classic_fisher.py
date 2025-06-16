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
device = "cuda:2" if torch.cuda.is_available() else "cpu"

# 普通CNN网络定义
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(32 * 7 * 7, 128)  # 假设输入是28x28，经过两次池化后为7x7
        self.fc2 = nn.Linear(128, num_classes)
        
    def forward(self, x):
        if x.shape[1] == 256:
            x = x.view(-1, 1, 16, 16)  # 16x16=256
            x = F.interpolate(x, size=(28, 28), mode='bilinear', align_corners=False)
        else:
            x = x.view(-1, 1, 28, 28)
        
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)  # 28x28 -> 14x14
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)  # 14x14 -> 7x7
        
        x = x.view(-1, 32 * 7 * 7)
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
# 定义自定义EWC（Elastic Weight Consolidation）策略
# ----------------------------------------------------------------------------- 

class CustomEWC(EWC):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fisher_matrices = {}
        self.optimal_weights = {}

    def update_fisher_information(self, experience):
        fisher_matrix = {}
        for name, param in self.model.named_parameters():
            fisher_matrix[name] = torch.zeros_like(param)

        self.model.eval()
        for sample, target in experience.dataset:
            self.optimizer.zero_grad()
            output = self.model(sample)
            loss = self.criterion(output, target)
            loss.backward()

            # Store second-order derivatives (i.e., Fisher Information)
            for name, param in self.model.named_parameters():
                fisher_matrix[name] += param.grad ** 2 / len(experience.dataset)

        self.fisher_matrices[experience.current_experience] = fisher_matrix
        self.optimal_weights[experience.current_experience] = {
            name: param.clone() for name, param in self.model.named_parameters()
        }

    def penalty_loss(self, experience):
        penalty = 0
        for name, param in self.model.named_parameters():
            fisher_matrix = self.fisher_matrices.get(experience.current_experience)
            optimal_weight = self.optimal_weights.get(experience.current_experience)
            if fisher_matrix is not None and optimal_weight is not None:
                penalty += (fisher_matrix[name] * (param - optimal_weight[name]) ** 2).sum()
        return penalty

    def train(self, experience):
        super().train(experience)
        # Update Fisher Information after each experience
        self.update_fisher_information(experience)

    def eval(self, experience):
        penalty = self.penalty_loss(experience)
        return super().eval(experience) + penalty


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
# 使用 EWC 持续学习策略
# ----------------------------------------------------------------------------- 
strategy = CustomEWC(
    model=model,
    optimizer=optimizer,
    criterion=criterion,
    ewc_lambda=1,
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
    with open(os.path.join(save_dir, f"splitmnist_EWC_classic_fisher_interim_results_exp_{experience.current_experience}.pkl"), "wb") as f:
        pickle.dump(task_accuracies, f)

# 保存最终结果
with open(os.path.join(save_dir, "splitmnist_EWC_classic_fisher_final.pkl"), "wb") as f:
    pickle.dump(task_accuracies, f)

print("✔ Training and evaluation completed!")
