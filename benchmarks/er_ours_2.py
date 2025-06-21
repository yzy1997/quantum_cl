#!/usr/bin/env python
# coding: utf-8

import torch
import torch.nn as nn
import torch.optim as optim
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
import time
import warnings

# 忽略一些不重要的警告
warnings.filterwarnings("ignore", category=UserWarning, module="torch.optim.lr_scheduler")
warnings.filterwarnings("ignore", category=UserWarning, module="pennylane")
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

# 量子电路参数
n_qubits = 8  # 8个量子比特可以编码256维向量 (2^8=256)
q_depth = 10  # 量子深度
q_delta = 0.1  # 参数初始化范围
train_epochs = 20  # 每个经验的训练轮次

# 量子计算设备设置
dev = qml.device("lightning.qubit", wires=n_qubits)

# 量子电路定义
def H_layer(n):
    """应用Hadamard门到所有量子比特"""
    for idx in range(n):
        qml.Hadamard(wires=idx)

def RY_layer(theta):
    """应用RY旋转门到所有量子比特"""
    for idx in range(n_qubits):
        qml.RY(theta[idx], wires=idx)

def RX_layer(theta):
    """应用RX旋转门到所有量子比特"""
    for idx in range(n_qubits):
        qml.RX(theta[idx], wires=idx)

def phase_layer(n):
    """应用相位门到量子比特"""
    for idx in range(n):
        if idx % 2 == 0:
            qml.S(wires=idx)
        else:
            qml.T(wires=idx)

def entangling_layer_3(n):
    """纠缠层"""
    for i in range(0, n - 1, 2):
        qml.CZ(wires=[i, i + 1])
    for i in range(1, n - 1, 2):
        qml.CZ(wires=[i, i + 1])

# 修改后的量子网络定义 - 使用Amplitude Embedding
@qml.qnode(dev, interface="torch", diff_method="parameter-shift")
def quantum_net_3(input_vector, q_weights_flat):
    # 确保输入是Tensor
    if not isinstance(input_vector, torch.Tensor):
        input_vector = torch.tensor(input_vector, dtype=torch.float32, requires_grad=True)
    
    # 确保权重是Tensor
    if not isinstance(q_weights_flat, torch.Tensor):
        q_weights_flat = torch.tensor(q_weights_flat, dtype=torch.float32, requires_grad=True)
    
    # 振幅编码 - 需要归一化的输入向量
    input_vector = input_vector / torch.norm(input_vector)  # 归一化
    qml.AmplitudeEmbedding(features=input_vector, wires=range(n_qubits), normalize=False)
    
    # 应用初始门
    H_layer(n_qubits)
    
    # 重塑权重
    q_weights = q_weights_flat.reshape(q_depth, n_qubits)
    
    # 应用量子层
    for layer in range(q_depth):
        RX_layer(q_weights[layer])  # 使用权重控制RX旋转
        phase_layer(n_qubits)
        entangling_layer_3(n_qubits)
    
    # 测量
    return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

# 可视化量子电路
print("量子电路结构:")
qml.drawer.use_style("pennylane")
fig, ax = qml.draw_mpl(quantum_net_3)(torch.randn(256), q_delta * torch.randn(q_depth * n_qubits))
plt.savefig("quantum_circuit.png")
plt.close()

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"使用设备: {device}")

# 量子-经典混合网络 - 针对256维输入和振幅编码优化
class AmplitudeQuantumNet(nn.Module):
    def __init__(self, n_qubits, q_depth, q_delta, num_classes=10):
        super().__init__()
        # 预处理网络 - 输出256维向量用于振幅编码
        self.pre_net = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 28x28 -> 14x14
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 14x14 -> 7x7
            
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 256),  # 输出256维向量
            nn.Tanh()  # 使用Tanh限制范围
        )
        
        # 量子参数 - 深度为10
        self.q_params = torch.nn.Parameter(q_delta * torch.randn(q_depth * n_qubits, requires_grad=True))
        
        # 后处理网络
        self.post_net = nn.Sequential(
            nn.Linear(n_qubits, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes)
        )
        
        # 初始化权重
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # 特征提取 - 输出256维向量
        features = self.pre_net(x)
        
        # 量子计算 - 处理批处理
        batch_size = x.size(0)
        quantum_out = []
        
        # 对每个样本单独处理
        for i in range(batch_size):
            # 获取当前样本的特征向量
            input_vector = features[i]
            
            # 执行量子电路
            q_out = quantum_net_3(input_vector, self.q_params)
            q_out_tensor = torch.tensor(q_out, device=device, dtype=torch.float32)
            quantum_out.append(q_out_tensor)
        
        quantum_out = torch.stack(quantum_out)
        
        # 分类
        return self.post_net(quantum_out)

# 数据预处理
transform = transforms.Compose([
    # transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))  # MNIST的标准归一化
])

# 创建SplitMNIST基准
benchmark = SplitMNIST(
    n_experiences=5,
    return_task_id=False,
    train_transform=transform,
    eval_transform=transform
)

print(f"训练经验数量: {len(benchmark.train_stream)}")
print(f"测试经验数量: {len(benchmark.test_stream)}")

# 创建模型
model = AmplitudeQuantumNet(n_qubits, q_depth, q_delta, num_classes=10).to(device)
print(f"模型参数数量: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

# 优化器和损失函数
optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-4)
criterion = nn.CrossEntropyLoss()

# 评估设置 - 移除了 suppress_warnings 参数
interactive_logger = InteractiveLogger()
eval_plugin = EvaluationPlugin(
    accuracy_metrics(minibatch=False, epoch=True, experience=True, stream=True),
    loss_metrics(minibatch=False, epoch=True, experience=True, stream=True),
    timing_metrics(epoch=True),
    forgetting_metrics(experience=True, stream=True),
    loggers=[interactive_logger]
)

# 经验回放插件
replay_plugin = ReplayPlugin(
    mem_size=1000,
    batch_size=16,
    batch_size_mem=16,
    task_balanced_dataloader=True
)

# 训练策略 - 20个轮次
strategy = Naive(
    model=model,
    optimizer=optimizer,
    criterion=criterion,
    train_epochs=20,  # 20个轮次
    device=device,
    evaluator=eval_plugin,
    plugins=[replay_plugin],
    train_mb_size=16
)

# 训练和评估
save_dir = "/home/yangz2/code/quantum_cl/results/list"
os.makedirs(save_dir, exist_ok=True)

task_accuracies = []
print("Starting training...")

for experience in benchmark.train_stream:
    print(f"\n--- Training on experience {experience.current_experience} ---")
    print(f"Classes in this experience: {experience.classes_in_this_experience}")
    print(f"Number of samples: {len(experience.dataset)}")
    
    # 训练当前经验
    strategy.train(experience)
    
    print(f"--- Evaluating after experience {experience.current_experience} ---")
    results = strategy.eval(benchmark.test_stream)
    task_accuracies.append(results)
    
    # 保存中间结果
    with open(os.path.join(save_dir, f"splitmnist_er_ours_qbit{n_qubits}_qdepth{q_depth}_tepoch{train_epochs}_interim_results_exp_{experience.current_experience}.pkl"), "wb") as f:
        pickle.dump(task_accuracies, f)
    
    # 打印当前准确率
    acc_key = 'Top1_Acc_Stream/eval_phase/test_stream/Task000'
    if acc_key in results:
        acc = results[acc_key]
        print(f"Global accuracy after experience {experience.current_experience}: {acc*100:.2f}%")
    else:
        # 列出所有键以便调试
        print(f"Error: Accuracy key not found. Available keys: {list(results.keys())}")
        # 尝试找到类似的键
        for key in results.keys():
            if "Acc_Stream" in key:
                acc = results[key]
                print(f"Found alternative accuracy key: {key} = {acc*100:.2f}%")
                break
        else:
            acc = 0.0
            print("No accuracy metric found in results")

# 保存最终结果
with open(os.path.join(save_dir, f"splitmnist_er_ours_qbit{n_qubits}_qdepth{q_depth}_tepoch{train_epochs}.pkl"), "wb") as f:
    pickle.dump(task_accuracies, f)

print("✔ Training and evaluation completed!")

# 保存完整模型
torch.save(model.state_dict(), os.path.join(save_dir, "final_model.pth"))
print("Model saved successfully.")