#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torchvision import transforms
from avalanche.benchmarks.classic import SplitMNIST
from avalanche.training import Naive
from avalanche.training.plugins import GEMPlugin, EWCPlugin, EvaluationPlugin, LRSchedulerPlugin
from avalanche.training.plugins import EvaluationPlugin
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

# 设置随机种子确保可复现性
torch.manual_seed(42)

# ================== 修改后的量子电路参数 ==================
n_qubits = 10  # 10个量子比特可以表示1024维状态 (2^10=1024)，足以处理784维输入
q_depth = 3    # 简化的量子深度（原为10）
q_delta = 0.5  # 参数初始化范围（增大以改善梯度）
train_epochs = 20  # 每个经验的训练轮次

# 量子计算设备设置
dev = qml.device("lightning.qubit", wires=n_qubits)

# ================== 修改后的量子电路定义 ==================
def entangling_layer(n):
    """简化的线性链式纠缠层"""
    for i in range(n-1):
        qml.CZ(wires=[i, i+1])

@qml.qnode(dev, interface="torch", diff_method="parameter-shift")
def quantum_net_angle(input_features, q_weights):
    """
    使用角度编码的量子电路
    输入: 
      input_features - 长度为n_qubits的特征向量
      q_weights - 量子权重参数 (q_depth, n_qubits)
    """
    # 角度编码 - 将经典特征映射到量子旋转角度
    for i in range(n_qubits):
        qml.RY(input_features[i] * np.pi, wires=i)  # 缩放特征到[0, π]范围
    
    # 应用量子层
    for layer in range(q_depth):
        # 参数化旋转
        for i in range(n_qubits):
            qml.RY(q_weights[layer, i], wires=i)
        
        # 简化的纠缠层
        entangling_layer(n_qubits)
    
    # 测量
    return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

# 可视化量子电路
print("量子电路结构:")
qml.drawer.use_style("pennylane")
# 使用784维输入进行可视化
fig, ax = qml.draw_mpl(quantum_net_angle)(torch.randn(n_qubits), torch.randn(q_depth, n_qubits))
plt.savefig("/home/yangz2/code/quantum_cl/results/figs/circuit3_angle.png")
plt.close()

device = "cuda:1" if torch.cuda.is_available() else "cpu"
print(f"使用设备: {device}")

# ================== 修改后的量子-经典混合网络 ==================
class AngleEncodingQuantumNet(nn.Module):
    def __init__(self, n_qubits, q_depth, q_delta, num_classes=10):  # 固定为10类
        """
        初始化:
          num_classes - 初始类别数（设置为10）
        """
        super().__init__()
        self.n_qubits = n_qubits
        self.q_depth = q_depth
        
        # 预处理网络 - 输出784维向量
        self.pre_net = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 28x28 -> 14x14
            
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 14x14 -> 7x7
            
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, 784),  # 输出784维向量
            nn.Tanh()  # 限制在[-1, 1]范围
        )
        
        # 特征选择层 - 从784维中选择最重要的n_qubits个特征
        self.feature_selector = nn.Linear(784, n_qubits)
        
        # 量子参数
        self.q_params = nn.Parameter(q_delta * torch.randn(q_depth, n_qubits))
        
        # 后处理网络 - 固定为10个输出
        self.post_net = nn.Sequential(
            nn.Linear(n_qubits, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes)
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
        x = x.to(device)
        # 特征提取 - 输出784维向量
        features = self.pre_net(x)
        
        # 选择最重要的n_qubits个特征
        selected_features = self.feature_selector(features)
        
        # 量子计算 - 处理批处理
        batch_size = x.size(0)
        quantum_out = []
        
        # 对每个样本单独处理
        for i in range(batch_size):
            # 获取当前样本的特征向量
            input_vector = selected_features[i]
            
            # 执行量子电路
            q_out = quantum_net_angle(input_vector, self.q_params)
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

# 创建SplitMNIST基准 - 使用全局类别标签
benchmark = SplitMNIST(
    n_experiences=5,
    return_task_id=False,
    train_transform=transform,
    eval_transform=transform,
    class_ids_from_zero_in_each_exp=False  # 使用全局类别标签
)

print(f"训练经验数量: {len(benchmark.train_stream)}")
print(f"测试经验数量: {len(benchmark.test_stream)}")

# 创建模型 - 固定为10个输出类
model = AngleEncodingQuantumNet(n_qubits, q_depth, q_delta, num_classes=10).to(device)
print(f"模型参数数量: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

# 优化器和损失函数
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)  # 增大学习率
criterion = nn.CrossEntropyLoss().to(device)  # 使用CrossEntropyLoss

# 评估设置
interactive_logger = InteractiveLogger()
eval_plugin = EvaluationPlugin(
    accuracy_metrics(minibatch=True, epoch=True, experience=True, stream=True),
    loss_metrics(minibatch=True, epoch=True, experience=True, stream=True),
    timing_metrics(epoch=True, epoch_running=True),
    cpu_usage_metrics(experience=True),
    forgetting_metrics(experience=True, stream=True),
    confusion_matrix_metrics(num_classes=10, save_image=True, stream=True),
    disk_usage_metrics(minibatch=True, epoch=True, experience=True, stream=True),
    loggers=[interactive_logger]
)

# 6. 配置GEM策略（调优参数）
gem_plugin = GEMPlugin(
    patterns_per_experience=300,  # 每个任务保留300个样本
    memory_strength=0.3          # 降低约束强度（更宽松）
)

# 7. 使用在线模式的EWC正则化插件
ewc_plugin = EWCPlugin(
    ewc_lambda=0.4,              # EWC正则化强度
    mode="online",               # 必须设置为online才能使用decay_factor
    decay_factor=0.9,            # Fisher矩阵衰减系数
    keep_importance_data=True    # 保留重要性数据用于后续任务
)

# 8. 添加学习率调度器
lr_scheduler = StepLR(optimizer, step_size=3, gamma=0.7)
lr_plugin = LRSchedulerPlugin(lr_scheduler)

# 训练策略
strategy = Naive(
    model,
    optimizer,
    criterion,
    train_mb_size=128,           # 训练批次大小
    eval_mb_size=256,            # 评估批次大小
    train_epochs=15,             # 增加训练轮数
    plugins=[gem_plugin, ewc_plugin, lr_plugin],
    evaluator=eval_plugin,
    device=device,
    # eval_every=1                 # 每个epoch后都评估
)

# 训练和评估
save_dir = "/home/yangz2/code/quantum_cl/results/list"
os.makedirs(save_dir, exist_ok=True)

task_accuracies = []
print("Starting training...")

for experience_idx, experience in enumerate(benchmark.train_stream):
    print(f"\n--- Training on experience {experience.current_experience} ---")
    print(f"Classes in this experience: {experience.classes_in_this_experience}")
    print(f"Number of samples: {len(experience.dataset)}")
    
    # 训练当前经验
    strategy.train(experience)
    
    print(f"--- Evaluating after experience {experience.current_experience} ---")
    results = strategy.eval(benchmark.test_stream)
    task_accuracies.append(results)
    
    # 保存中间结果
    with open(os.path.join(save_dir, f"splitmnist_GEM_ours_qbit{n_qubits}_qdepth{q_depth}_tepoch{train_epochs}_interim_results_exp_{experience.current_experience}.pkl"), "wb") as f:
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


# In[ ]:


# 保存最终结果
with open(os.path.join(save_dir, f"splitmnist_GEM_ours_qbit{n_qubits}_qdepth{q_depth}_tepoch{train_epochs}.pkl"), "wb") as f:
    pickle.dump(task_accuracies, f)

print("✔ Training and evaluation completed!")

