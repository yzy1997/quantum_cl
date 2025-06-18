#!/usr/bin/env python
# coding: utf-8

# In[1]:


#!/usr/bin/env python
# coding: utf-8

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pickle, os
from avalanche.benchmarks.classic import SplitMNIST
from avalanche.training import Naive
from avalanche.training.plugins import ReplayPlugin
from avalanche.training.plugins import EvaluationPlugin
from avalanche.evaluation.metrics import (
    accuracy_metrics, loss_metrics,
    forgetting_metrics, timing_metrics,
    cpu_usage_metrics, disk_usage_metrics
)
from avalanche.logging import InteractiveLogger

# 设备切到 cuda:2
device = "cuda:2" if torch.cuda.is_available() else "cpu"

# 定义网络（不变）
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, 1, 1)
        self.conv2 = nn.Conv2d(16, 32, 3, 1, 1)
        self.fc1   = nn.Linear(32 * 7 * 7, 128)
        self.fc2   = nn.Linear(128, num_classes)
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = x.view(-1, 32 * 7 * 7)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

# 基准
benchmark = SplitMNIST(n_experiences=5, return_task_id=False)

# 模型、优化器、损失
model     = SimpleCNN(10).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

# 日志与评估插件（保留你原来的配置）
interactive_logger = InteractiveLogger()
eval_plugin = EvaluationPlugin(
    accuracy_metrics(minibatch=True, epoch=True, experience=True, stream=True),
    loss_metrics(minibatch=True,     epoch=True, experience=True, stream=True),
    timing_metrics(epoch=True, epoch_running=True),
    cpu_usage_metrics(experience=True),
    forgetting_metrics(experience=True, stream=True),
    disk_usage_metrics(minibatch=True, epoch=True, experience=True, stream=True),
    loggers=[interactive_logger]
)

# ==== 关键修改：使用 Naive + ReplayPlugin 代替 EWC ====
# Experience Replay 插件，最多存 200 张图，batch_size=32
replay_plugin = ReplayPlugin(
    mem_size=200,
    batch_size=32,      # 每个 minibatch 中新样本的数量
    batch_size_mem=32,  # 每个 minibatch 中重放样本的数量
    task_balanced_dataloader=False
)

strategy = Naive(
    model=model,
    optimizer=optimizer,
    criterion=criterion,
    train_epochs=20,
    device=device,
    evaluator=eval_plugin,
    plugins=[replay_plugin]
)
# ======================================================

# 训练 & 评估主循环（略微改了路径）
save_dir = "/home/yangz2/code/quantum_cl/results/list_er"
os.makedirs(save_dir, exist_ok=True)

task_accuracies = []
print("Starting ER-based training...")
for experience in benchmark.train_stream:
    print(f"\n--- Training on experience {experience.current_experience} ---")
    strategy.train(experience)

    print(f"--- Evaluating after experience {experience.current_experience} ---")
    results = strategy.eval(benchmark.test_stream)
    task_accuracies.append(results)

    with open(os.path.join(
        save_dir,
        f"splitmnist_ER_cnn_interim_exp_{experience.current_experience}.pkl"
    ), "wb") as f:
        pickle.dump(task_accuracies, f)

with open(os.path.join(save_dir, "splitmnist_ER_cnn_final.pkl"), "wb") as f:
    pickle.dump(task_accuracies, f)

print("✔ ER Training and evaluation completed!")

