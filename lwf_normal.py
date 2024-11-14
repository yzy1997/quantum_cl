#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import urllib.request
import zipfile
import torch
from torch import nn
from torch.optim import SGD
from torchvision import models, transforms
from torch.utils.data import random_split, DataLoader
from torchvision.datasets import ImageFolder
from avalanche.benchmarks import dataset_benchmark
from avalanche.training.plugins import LwFPlugin
from avalanche.training import Naive
from avalanche.evaluation.metrics import accuracy_metrics, loss_metrics, timing_metrics
from avalanche.logging import InteractiveLogger
from avalanche.training.plugins import EvaluationPlugin
import matplotlib.pyplot as plt

# 1. Download Tiny ImageNet dataset and extract it
def download_tiny_imagenet(data_dir):
    url = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"
    zip_file = os.path.join(data_dir, "tiny-imagenet-200.zip")

    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    if not os.path.exists(zip_file):
        print("Downloading Tiny ImageNet dataset...")
        urllib.request.urlretrieve(url, zip_file)

    extracted_dir = os.path.join(data_dir, "tiny-imagenet-200")
    if not os.path.exists(extracted_dir):
        print("Extracting dataset...")
        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
            zip_ref.extractall(data_dir)

    return extracted_dir

data_dir = "./data"
tiny_imagenet_dir = './data/tiny-imagenet-200'

# Organize validation set (if needed, we will move the images based on their annotations)
val_dir = os.path.join(tiny_imagenet_dir, 'val')
val_images_dir = os.path.join(val_dir, 'images')
val_annotations_file = os.path.join(val_dir, 'val_annotations.txt')

# Create class subdirectories in the validation folder
def organize_val_images(val_dir, val_images_dir, val_annotations_file):
    if not os.path.exists(val_images_dir):
        return  # Already organized

    print("Organizing validation images into class folders...")
    with open(val_annotations_file, 'r') as f:
        for line in f.readlines():
            parts = line.strip().split('\t')
            img_file, class_id = parts[0], parts[1]
            class_dir = os.path.join(val_dir, class_id)
            if not os.path.exists(class_dir):
                os.makedirs(class_dir)
            
            # Move images into corresponding class folder
            img_src_path = os.path.join(val_images_dir, img_file)
            img_dst_path = os.path.join(class_dir, img_file)
            if os.path.exists(img_src_path):
                os.rename(img_src_path, img_dst_path)

    # Remove the original images folder
    if os.path.exists(val_images_dir):
        os.rmdir(val_images_dir)

organize_val_images(val_dir, val_images_dir, val_annotations_file)

# 2. Define transformations and load Tiny ImageNet data using ImageFolder
transform = transforms.Compose([
    transforms.Resize(64),  # Tiny ImageNet images are 64x64
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load train and validation datasets
train_dataset = ImageFolder(root=os.path.join(tiny_imagenet_dir, 'train'), transform=transform)
val_dataset = ImageFolder(root=os.path.join(tiny_imagenet_dir, 'val'), transform=transform)

# 3. 将 train_dataset 分为 50 个 experience
num_experiences = 50
dataset_len = len(train_dataset)
subset_size = dataset_len // num_experiences

# random_split 用于将 train_dataset 划分为 num_experiences 个子集
train_subsets = random_split(train_dataset, [subset_size] * (num_experiences - 1) + [dataset_len - subset_size * (num_experiences - 1)])

# 使用 dataset_benchmark 创建 50 个 experience 的基准
benchmark = dataset_benchmark(
    train_datasets=train_subsets,  # 切分后的 50 个子集
    test_datasets=[val_dataset] * num_experiences  # 每个 experience 使用相同的验证集
)

# 4. Load pre-trained ResNet18 and modify the final layer for 200 classes
resnet18 = models.resnet18(pretrained=True)
# resnet18.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)  # Change input channels to 1
num_classes = 200
resnet18.fc = nn.Linear(resnet18.fc.in_features, num_classes)

# 5. Define optimizer
optimizer = SGD(resnet18.parameters(), lr=0.01)

# 6. Define loss function
criterion = nn.CrossEntropyLoss()

# 7. Define evaluation plugin for logging metrics
eval_plugin = EvaluationPlugin(
    accuracy_metrics(experience=True, stream=True),
    loss_metrics(minibatch=True, experience=True),
    timing_metrics(experience=True, stream=True),
    loggers=[InteractiveLogger()]
)




# In[ ]:


###############################################LwF Strategy###############################################
from avalanche.training import LwF

# 1. 定义 LwF 策略，直接使用 LwF 类
strategy = LwF(
    model=resnet18,
    optimizer=optimizer,
    criterion=criterion,
    alpha=0.5,  # 平衡新任务和旧任务损失的权重
    temperature=2.0,  # 蒸馏温度，越高的值越软化
    train_mb_size=32,
    train_epochs=4,
    eval_mb_size=100,
    evaluator=eval_plugin,
    device='cuda' if torch.cuda.is_available() else 'cpu'
)

# Save training accuracy and loss history for plotting
accuracy_history = []
loss_history = []

# 9. Training and evaluation loop
# Use experiences from Avalanche's benchmark, such as from benchmark.train_stream
for epoch in range(30):  # 训练 4 个 epoch
    print(f"Training epoch {epoch}")
    
    # Iterate over experiences from the benchmark's training stream
    for experience in benchmark.train_stream:
        # Train on the current experience
        strategy.train(experience)
    
    # 在每个 epoch 后进行评估
    results = strategy.eval(benchmark.test_stream)
    # print('*************',results.keys())
    
    # Save accuracy and loss
    for key in results.keys():
        if 'Top1_Acc_Exp' in key:  # 寻找包含 Top1_Acc_Exp 的键
            accuracy_history.append(results[key])
            print(f"Added accuracy for {key}: {results[key]}")
        if 'Loss_Exp' in key:  # 寻找包含 Loss_Exp 的键
            loss_history.append(results[key])
            print(f"Added loss for {key}: {results[key]}")

# 10. Plot accuracy and loss over epochs
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
plt.savefig('lwf_tiny_imagenet_LWF.png')
plt.show()

