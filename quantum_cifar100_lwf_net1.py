# Import required libraries
import torch
from torch.nn import CrossEntropyLoss, Linear
from torch.optim import SGD
from torchvision.transforms import Normalize, Compose, ToTensor
from avalanche.benchmarks.classic import SplitCIFAR100
from avalanche.training.plugins import LwFPlugin
from avalanche.training import LwF
from avalanche.evaluation.metrics import accuracy_metrics, loss_metrics, timing_metrics
from avalanche.logging import InteractiveLogger
from avalanche.training.plugins import EvaluationPlugin
from torch.utils.data import DataLoader
from torch import nn
import matplotlib.pyplot as plt

# Configuration parameters
batch_size = 256
num_epochs = 10
train_mb_size = 256
eval_mb_size = 100
alpha = 0.5
temperature = 2.0
step = 0.0004
device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

# **Step 1: Configure CIFAR-100 as SplitCIFAR100 dataset**
# Data preprocessing
transform = Compose([
    ToTensor(),
    Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # CIFAR-100 normalization
])

# Create SplitCIFAR100 benchmark
benchmark = SplitCIFAR100(
    n_experiences=10,  # Split CIFAR-100 into 10 experiences
    seed=1234,
    return_task_id=False,
    train_transform=transform,
    eval_transform=transform
)

# **Step 2: Define a hybrid model**
# Use ResNet18 as feature extractor
from torchvision.models import resnet18

resnet_model = resnet18(pretrained=True)
for param in resnet_model.parameters():
    param.requires_grad = False  # Freeze pre-trained layers

# Custom quantum-inspired layer (DressedQuantumNet)
class DressedQuantumNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.pre_net = nn.Linear(512, 12)  # Map to quantum input size
        self.q_params = nn.Parameter(torch.randn(10 * 12) * 0.01)  # Random quantum weights
        self.post_net = nn.Linear(12, 100)  # CIFAR-100 has 100 classes

    def forward(self, input_features):
        pre_out = self.pre_net(input_features)
        q_out = torch.tanh(pre_out) * torch.pi / 2.0
        return self.post_net(q_out)

# Replace the classification layer of ResNet with DressedQuantumNet
resnet_model.fc = DressedQuantumNet()

# Place the model on the appropriate device
model_hybrid = resnet_model.to(device)

# **Step 3: Define the training strategy**
criterion = CrossEntropyLoss()
optimizer_hybrid = SGD(model_hybrid.parameters(), lr=step, momentum=0.9)

# Define evaluation plugin for logging
eval_plugin = EvaluationPlugin(
    accuracy_metrics(epoch=True, experience=True, stream=True),
    loss_metrics(epoch=True, experience=True, stream=True),
    timing_metrics(experience=True, stream=True),
    loggers=[InteractiveLogger()]
)

# Define LwF strategy
strategy = LwF(
    model=model_hybrid,
    optimizer=optimizer_hybrid,
    criterion=criterion,
    alpha=alpha,
    temperature=temperature,
    train_mb_size=train_mb_size,
    train_epochs=num_epochs,
    eval_mb_size=eval_mb_size,
    evaluator=eval_plugin,
    device=device
)

# **Step 4: Training and evaluation**
accuracy_history = []
loss_history = []

for epoch in range(num_epochs):
    print(f"Training epoch {epoch}")
    
    for experience in benchmark.train_stream:
        print(f"Training on experience {experience.current_experience}")
        strategy.train(experience)  # Train on the current experience
    
    # Evaluate after each epoch
    results = strategy.eval(benchmark.test_stream)
    
    # Log accuracy and loss for plotting
    for key in results.keys():
        if 'Top1_Acc_Exp' in key:  # Collect accuracy
            accuracy_history.append(results[key])
            print(f"Added accuracy for {key}: {results[key]}")
        if 'Loss_Exp' in key:  # Collect loss
            loss_history.append(results[key])
            print(f"Added loss for {key}: {results[key]}")

# **Step 5: Plot accuracy and loss**
plt.figure(figsize=(10, 5))

# Plot accuracy
plt.subplot(1, 2, 1)
plt.plot(accuracy_history, label='Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Accuracy over epochs')
plt.legend()

# Plot loss
plt.subplot(1, 2, 2)
plt.plot(loss_history, label='Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss over epochs')
plt.legend()

plt.tight_layout()
plt.savefig('./results/lwf_split_cifar100.png')
plt.show()
