import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import torchvision.transforms as transforms
import os
import pickle

# 导入Avalanche库
from avalanche.benchmarks.classic import SplitMNIST
from avalanche.training.plugins import EvaluationPlugin, LRSchedulerPlugin
from avalanche.training.supervised import ER_ACE
from avalanche.evaluation.metrics import accuracy_metrics, forgetting_metrics, loss_metrics, timing_metrics, cpu_usage_metrics, confusion_matrix_metrics, disk_usage_metrics
from avalanche.logging import InteractiveLogger

# 设置随机种子确保可复现性
torch.manual_seed(42)

# 1. 设置设备 - 使用cuda:2
device = torch.device("cuda:2")
print(f"Using device: {device}")

# 2. 增强型CNN模型定义
class EnhancedCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(EnhancedCNN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.fc_layers = nn.Sequential(
            nn.Linear(128 * 3 * 3, 512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, num_classes)
        )
        
    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        return self.fc_layers(x)

# 3. 修复数据变换问题 - 确保正确的数据顺序
# 创建基准数据集
benchmark = SplitMNIST(
    n_experiences=5,  # 分成5个任务
    return_task_id=False,
    # 移除数据增强以简化问题
    train_transform=transforms.Compose([
        # transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ]),
    eval_transform=transforms.Compose([
        # transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
)

# 4. 初始化模型、优化器和损失函数
model = EnhancedCNN(num_classes=10).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
criterion = nn.CrossEntropyLoss()

# 5. 设置评估插件和日志记录
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



# 8. 添加学习率调度器
lr_scheduler = StepLR(optimizer, step_size=3, gamma=0.7)
lr_plugin = LRSchedulerPlugin(lr_scheduler)

# 9. 创建持续学习策略
strategy = ER_ACE(
    model,
    optimizer,
    criterion,
    train_mb_size=128,           # 训练批次大小
    eval_mb_size=256,            # 评估批次大小
    train_epochs=15,             # 增加训练轮数
    plugins=[lr_plugin],
    evaluator=eval_plugin,
    device=device,
    # eval_every=1                 # 每个epoch后都评估
)

# 10. 训练和评估循环
task_accuracies = []

print("Starting training on device:", device)
for experience in benchmark.train_stream:
    print(f"\n--- Training on experience {experience.current_experience} ---")
    print(f"Classes: {experience.classes_in_this_experience}")
    
    # 训练当前任务 - 移除num_workers参数
    strategy.train(experience)
    
    print("--- Evaluating on test stream ---")
    # 在整个测试流上评估
    results = strategy.eval(benchmark.test_stream)
    
    # 收集结果
    task_accuracies.append(results)
    
    # 打印当前性能
    print(f"\nAfter experience {experience.current_experience}:")
    for k, v in results.items():
        if 'Top1_Acc_Exp' in k:
            task_id = k.split('/')[-2]
            print(f"\tAccuracy on Task {task_id}: {v*100:.2f}%")
        elif 'Loss_Exp' in k:
            task_id = k.split('/')[-2]
            print(f"\tLoss on Task {task_id}: {v:.4f}")
    
    # 打印遗忘情况
    if experience.current_experience > 0:
        print(f"\tAverage Forgetting: {results['StreamForgetting/eval_phase/test_stream']*100:.2f}%")

# 11. 保存结果到文件
file_path = "/home/yangz2/code/quantum_cl/results/list/splitmnist_GEM.pkl"

# 确保目录存在
os.makedirs(os.path.dirname(file_path), exist_ok=True)

# 存储到文件
with open(file_path, "wb") as f:
    pickle.dump(task_accuracies, f)

print(f"\nResults saved to {file_path}")

# 保存最终模型
torch.save(model.state_dict(), "splitmnist_gem_final.pth")
print("Model saved to splitmnist_gem_final.pth")

# 打印最终性能
print("\nFinal Performance Summary:")
final_metrics = task_accuracies[-1]
for k, v in final_metrics.items():
    if 'Acc_Stream' in k or 'Forgetting' in k or 'Loss_Stream' in k:
        print(f"{k}: {v}")