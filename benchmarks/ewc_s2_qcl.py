#!/usr/bin/env python
# coding: utf-8

# In[9]:


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import pennylane as qml
from pennylane.qnn import TorchLayer
from torchvision import transforms

from avalanche.benchmarks.classic import SplitMNIST
from avalanche.training import EWC
from avalanche.training.plugins import EvaluationPlugin,SupervisedPlugin
from avalanche.evaluation.metrics import forgetting_metrics, \
accuracy_metrics, loss_metrics, timing_metrics, cpu_usage_metrics, \
confusion_matrix_metrics, disk_usage_metrics
from avalanche.logging import InteractiveLogger
import pickle
import os
import numpy as np


# In[52]:


n_qubits = 8
n_layers = 4
dev = qml.device("lightning.qubit", wires=n_qubits)

def feature_encoding(inputs):
    qml.AmplitudeEmbedding(inputs, wires=list(range(n_qubits)), normalize=True)
        # 不是环形连接，只连接到下一个比特
    for i in range(0, n_qubits - 1, 2):
            qml.CNOT(wires=[i, i + 1])
    for i in range(1, n_qubits - 1, 2):
        qml.CNOT(wires=[i, i + 1])
            
def varitional_block(params):
    for i in range(n_qubits):
        qml.RX(params[0,i], wires=i)
        qml.RZ(params[1,i], wires=i)
        qml.RX(params[2,i], wires=i)
    for i in range(0, n_qubits - 1, 2):
            qml.CNOT(wires=[i, i + 1])
    for i in range(1, n_qubits - 1, 2):
        qml.CNOT(wires=[i, i + 1])
    
@qml.qnode(dev)
def quantum_circuit(inputs, weights):
    feature_encoding(inputs)
    for layer in range(n_layers):
        varitional_block(weights[layer])
    return qml.expval(qml.PauliZ(n_qubits // 2))  # 返回中间比特的期望值

qml.drawer.use_style("pennylane")
# 输入：8个量子比特
inputs = torch.randn(2** n_qubits)
# 权重：4 层，每层 8 比特，每比特三个参数（RX, RZ, RX）
weights = torch.randn(n_layers, 3, n_qubits)
# 绘制量子电路
fig, ax = qml.draw_mpl(quantum_circuit)(inputs, weights)
plt.show()
    


# In[54]:


# -----------------------------
# 2. TorchLayer + PyTorch Model
# -----------------------------
weight_shapes = {"weights": (n_layers, 3, n_qubits)}
qlayer = TorchLayer(quantum_circuit, weight_shapes)

# Initialize weights with smaller values
with torch.no_grad():
    qlayer.weights.data = qlayer.weights.data * 0.1

class QuantumClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_layer = qlayer
        self.output = nn.Linear(1, 10)

    def forward(self, x):
        # x: (batch, 784) → transform后是 (batch,256)
        # 1) 直接 normalize（AmplitudeEmbedding 自带归一化可省略）
        #    这里做均值方差标准化
        x = (x - x.mean(dim=1, keepdim=True)) / (x.std(dim=1, keepdim=True) + 1e-8)

        # 2) 量子层一次性接收 256 维
        x = self.q_layer(x)     # → shape (batch,)
        x = x.unsqueeze(-1)     # → (batch,1)

        # 3) 经典线性分类
        x = self.output(x)      # → (batch,10)
        return F.log_softmax(x, dim=1)


# In[55]:


# -----------------------------
# 3. Data Transform (Select 8 pixels)
# -----------------------------
def select_and_normalize(x):
    x = torch.tensor(np.array(x), dtype=torch.float32).view(-1)  # 784
    x = x[:256]             # 取前 256
    # 幅度编码会 normalize，这里只做简单缩放到 [-1,1]
    x = (x - x.mean()) / x.std()
    return x

transform = transforms.Compose([
    transforms.Lambda(select_and_normalize)
])

benchmark = SplitMNIST(n_experiences=5, return_task_id=False,
                       train_transform=transform, eval_transform=transform)


# In[56]:


# -----------------------------
# 4. Avalanche EWC Strategy Setup
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = QuantumClassifier().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.NLLLoss()

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

strategy = EWC(
    model=model,
    optimizer=optimizer,
    criterion=criterion,
    ewc_lambda=0.4,
    train_epochs=10,
    device=device,
    evaluator=eval_plugin
)

class GradientClipPlugin(SupervisedPlugin):
    def before_backward(self, strategy, **kwargs):
        torch.nn.utils.clip_grad_norm_(strategy.model.parameters(), max_norm=1.0)

strategy.plugins.append(GradientClipPlugin())


# In[57]:


# -----------------------------
# 5. Training & Evaluation
# -----------------------------
task_accuracies = []
save_dir = "/home/yangz2/code/quantum_cl/results"
os.makedirs(save_dir, exist_ok=True)

print("Starting training...")
for experience in benchmark.train_stream:
    print(f"\n--- Training on experience {experience.current_experience} ---")
    strategy.train(experience)
    
    print(f"--- Evaluating after experience {experience.current_experience} ---")
    results = strategy.eval(benchmark.test_stream)
    task_accuracies.append(results)
    
    # Save intermediate results
    with open(os.path.join(save_dir, f"interim_results_exp_{experience.current_experience}.pkl"), "wb") as f:
        pickle.dump(task_accuracies, f)

# Save final results
with open(os.path.join(save_dir, "splitmnist_EWC_s2_qbit8_qdepth4_tepoch10.pkl"), "wb") as f:
    pickle.dump(task_accuracies, f)


# In[ ]:


# # Load results
# with open(os.path.join(save_dir, "splitmnist_EWC_s2_qbit8_qdepth4_tepoch10.pkl"), "rb") as f:
#     results = pickle.load(f)


# In[ ]:


# import matplotlib.pyplot as plt

# # results = pickle.load(…)  # 已经加载好的 list of dict

# # 1) 准备横轴（experience 序号）
# exp_idxs = list(range(1, len(results) + 1))

# # 2) 从 results 中提取三条曲线
# accuracies  = [r['Top1_Acc_Stream/eval_phase/test_stream/Task000'] for r in results]
# losses      = [r['Loss_Stream/eval_phase/test_stream/Task000']      for r in results]
# forgettings = [r['StreamForgetting/eval_phase/test_stream']         for r in results]

# # 3) 画 Accuracy
# plt.figure()
# plt.plot(exp_idxs, accuracies)
# plt.xlabel('Experience')
# plt.ylabel('Accuracy')
# plt.title('Accuracy over Experiences')
# plt.xticks(exp_idxs)
# plt.savefig(os.path.join(save_dir, "splitmnist_EWC_s2_qbit8_qdepth4_tepoch10_acc.png"))

# # 4) 画 Loss
# plt.figure()
# plt.plot(exp_idxs, losses)
# plt.xlabel('Experience')
# plt.ylabel('Loss')
# plt.title('Loss over Experiences')
# plt.xticks(exp_idxs)
# plt.savefig(os.path.join(save_dir, "splitmnist_EWC_s2_qbit8_qdepth4_tepoch10_loss.png"))

# # 5) 画 Forgetting
# plt.figure()
# plt.plot(exp_idxs, forgettings)
# plt.xlabel('Experience')
# plt.ylabel('Forgetting')
# plt.title('Forgetting over Experiences')
# plt.xticks(exp_idxs)
# plt.savefig(os.path.join(save_dir, "splitmnist_EWC_s2_qbit8_qdepth4_tepoch10_forget.png"))

