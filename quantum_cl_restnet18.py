import torch
from torch.nn import CrossEntropyLoss, Linear
from torch.optim import SGD
from torchvision.models import resnet18
from torchvision import transforms # 用于数据预处理
from torchvision.transforms import Normalize, Compose, ToTensor
from avalanche.benchmarks.classic import SplitCIFAR100
from avalanche.training import ICaRL
from avalanche.training.plugins import EvaluationPlugin
from avalanche.evaluation.metrics import accuracy_metrics, loss_metrics, forgetting_metrics
from avalanche.logging import InteractiveLogger
import pickle   # 用于保存结果


# 数据预处理
transform = transforms.Compose([
    transforms.Resize(64),  # Tiny ImageNet images are 64x64
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 定义 buffer_transform 用于张量标准化
def buffer_transform(x):
    if isinstance(x, torch.Tensor):
        return (x - 0.5) / 0.5  # 张量标准化
    return x

# 创建 SplitCIFAR100 数据集
benchmark = SplitCIFAR100(
    n_experiences=10,  # 将 CIFAR-100 分为 10 个增量任务
    seed=1234,
    return_task_id=False,
    train_transform=transform,
    eval_transform=transform
)

# 定义 ResNet18 模型作为特征提取器
resnet_model = resnet18(pretrained=False)  # 不使用预训练权重
feature_extractor = torch.nn.Sequential(
    *list(resnet_model.children())[:-1],  # 移除分类层
    torch.nn.Flatten()  # 展平为 [batch_size, 512]
)
hidden_size = resnet_model.fc.in_features  # 特征提取器的输出维度

# 定义分类器
output_size = 100  # CIFAR-100 总类别数
classifier = Linear(hidden_size, output_size)

# 优化器和损失函数
optimizer = SGD(list(feature_extractor.parameters()) + list(classifier.parameters()), lr=0.01, momentum=0.9)
criterion = CrossEntropyLoss()

# 设置评估插件
eval_plugin = EvaluationPlugin(
    accuracy_metrics(epoch=True, experience=True, stream=True),
    loss_metrics(epoch=True, experience=True, stream=True),
    forgetting_metrics(stream=True),
    loggers=[InteractiveLogger()]
)

# 定义 ICaRL 策略
strategy = ICaRL(
    feature_extractor=feature_extractor,
    classifier=classifier,
    optimizer=optimizer,
    memory_size=2000,  # 存储池大小
    train_mb_size=64,
    eval_mb_size=64,
    train_epochs=1,
    device="cuda" if torch.cuda.is_available() else "cpu",
    evaluator=eval_plugin,
    buffer_transform=buffer_transform,  # 使用适合张量的标准化函数
    fixed_memory=False
)

# 训练和评估
results = []
for experience in benchmark.train_stream:
    print(f"Training on experience {experience.current_experience}")
    strategy.train(experience)  # 训练
    print(f"Evaluating on experience {experience.current_experience}")
    results.append(strategy.eval(benchmark.test_stream))  # 评估



# 存储到文件
with open("results/list/CIFAR100_ICaRL_cls_tepoch1.pkl", "wb") as f:
    pickle.dump(results, f) 