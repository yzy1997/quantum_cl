#!/usr/bin/env python
# coding: utf-8

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from avalanche.benchmarks.classic import SplitMNIST
from avalanche.training import Naive
from avalanche.training.plugins import EvaluationPlugin, ReplayPlugin
from avalanche.evaluation.metrics import forgetting_metrics, \
accuracy_metrics, loss_metrics, timing_metrics, cpu_usage_metrics, \
confusion_matrix_metrics, disk_usage_metrics
from avalanche.logging import InteractiveLogger
import pennylane as qml
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
from pennylane.qnn import TorchLayer

# 量子电路参数
n_qubits = 8
n_layers = 4
q_depth = 4
q_delta = 0.1

# 量子计算设备设置
dev = qml.device("lightning.qubit", wires=n_qubits)

# 量子电路定义
def phase_layer(w):
    """Layer of S gates and T gates to flip the phase of the qubits."""
    for idx in range(w):
        if idx % 2 == 0:
            qml.S(wires=idx)
        else:
            qml.T(wires=idx)

def entangling_layer_3(nqubits):
    """Layer of controlled-Rz gates followed by zz entangling gates."""
    params = np.random.uniform(0, 2*np.pi, (nqubits,))
    for i in range(0, nqubits - 1, 2):
        qml.ctrl(qml.RZ, control=i)(params[i], wires=i + 1)
    for i in range(1, nqubits - 1, 2):
        qml.IsingZZ(params[i], wires=[i, i + 1])

def measurement_layer(nqubits):
    """Measurement layer in the Z basis."""
    return [qml.expval(qml.PauliZ(i)) for i in range(nqubits)]

# 量子网络定义
@qml.qnode(dev, interface="torch", diff_method="parameter-shift")
def quantum_net_3(q_input_features, q_weights_flat):
    """Quantum circuit for the network."""
    if not isinstance(q_weights_flat, torch.Tensor):
        q_weights_flat = torch.tensor(q_weights_flat, dtype=torch.float32, requires_grad=True)

    q_weights = q_weights_flat.reshape(q_depth, n_qubits)

    # Apply quantum layers
    for _ in range(q_depth):
        phase_layer(n_qubits)
        entangling_layer_3(n_qubits)
    
    # Return Pauli-Z expectation values
    exp_val = [qml.expval(qml.PauliZ(position)) for position in range(n_qubits)]
    return exp_val

device = "cuda:1" if torch.cuda.is_available() else "cpu"

# 量子-经典混合网络 - 使用原始28x28维度输入
class DressedQuantumNet(nn.Module):
    def __init__(self, n_qubits, n_layers, q_depth, q_delta, num_classes=10):
        super().__init__()
        # 使用更强大的预处理网络处理原始28x28图像
        self.pre_net = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),  # 输入通道1，输出通道16
            nn.ReLU(),
            nn.MaxPool2d(2),  # 28x28 -> 14x14
            nn.Conv2d(16, 32, kernel_size=3, padding=1),  # 输入通道16，输出通道32
            nn.ReLU(),
            nn.MaxPool2d(2),  # 14x14 -> 7x7
            nn.Flatten(),  # 7x7x32 = 1568
            nn.Linear(1568, 256),  # 降维到256
            nn.ReLU(),
            nn.Linear(256, n_qubits)  # 最终输出量子比特数
        )
        self.q_params = torch.nn.Parameter(q_delta * torch.randn(q_depth * n_qubits, requires_grad=True))
        self.post_net = nn.Linear(n_qubits, num_classes)

    def forward(self, input_features):
        """Forward pass through the quantum-encoded network."""
        # 输入形状: (batch_size, 1, 28, 28)
        pre_out = self.pre_net(input_features.to(device))
        q_in = torch.tanh(pre_out) * np.pi / 2.0  # Map to quantum input range

        q_out = torch.empty((0, n_qubits), device=device, dtype=torch.float32, requires_grad=True)
        for elem in q_in:
            q_out_elem = torch.hstack(quantum_net_3(elem, self.q_params)).float().unsqueeze(0).to(device)
            q_out = torch.cat((q_out, q_out_elem), dim=0).requires_grad_(True)

        return self.post_net(q_out)

# 创建数据预处理转换
transform = transforms.Compose([
    # transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# 直接使用 Avalanche 的 SplitMNIST 创建 benchmark
benchmark = SplitMNIST(
    n_experiences=5,
    return_task_id=False,
    train_transform=transform,
    eval_transform=transform
)

# 打印benchmark信息
print("Number of experiences in train stream:", len(benchmark.train_stream))
print("Number of experiences in test stream:", len(benchmark.test_stream))

# -----------------------------------------------------------------------------
# 设置训练环境
# -----------------------------------------------------------------------------
model = DressedQuantumNet(n_qubits, n_layers, q_depth, q_delta, num_classes=10).to(device)
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
# 使用 Replay 持续学习策略
# -----------------------------------------------------------------------------
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
    with open(os.path.join(save_dir, f"splitmnist_er_ours_qbit8_qdepth4_tepoch10_interim_results_exp_{experience.current_experience}.pkl"), "wb") as f:
        pickle.dump(task_accuracies, f)

# 保存最终结果
with open(os.path.join(save_dir, "splitmnist_er_ours_qbit8_qdepth4_tepoch10.pkl"), "wb") as f:
    pickle.dump(task_accuracies, f)

print("✔ Training and evaluation completed!")