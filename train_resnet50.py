import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
from tqdm import tqdm
import numpy as np

# -------------------- 配置参数 --------------------
DATA_ROOT = ".。/EmoSet-118K"          # 数据集根目录
TRAIN_JSON = os.path.join(DATA_ROOT, "train.json")
VAL_JSON   = os.path.join(DATA_ROOT, "val.json")
# TEST_JSON  = os.path.join(DATA_ROOT, "test.json")   # 评估时使用

BATCH_SIZE = 64
EPOCHS = 50
LEARNING_RATE = 1e-4
NUM_CLASSES = 8
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_SAVE_PATH = "./weights/resnet50_emo8.pth"

# 情感类别映射（顺序固定，与 info.json 一致）
EMOTIONS = ["amusement", "awe", "contentment", "excitement",
            "anger", "disgust", "fear", "sadness"]
EMO2ID = {e: i for i, e in enumerate(EMOTIONS)}
ID2EMO = {i: e for e, i in EMO2ID.items()}

# -------------------- 数据集类 --------------------
class EmoSetDataset(Dataset):
    def __init__(self, samples, data_root, transform=None):
        """
        samples: 列表，每个元素为 [label_str, img_rel_path, ann_rel_path]
        data_root: 数据集根目录，用于拼接图片完整路径
        """
        self.samples = samples
        self.data_root = data_root
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        label_str, img_rel_path, _ = self.samples[idx]
        img_path = os.path.join(self.data_root, img_rel_path)   # 例如 ./data/EmoSet-118K/image/amusement/xxx.jpg
        image = Image.open(img_path).convert('RGB')
        label = EMO2ID[label_str]   # 直接根据字符串映射
        if self.transform:
            image = self.transform(image)
        return image, label

# -------------------- 数据增强与预处理 --------------------
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# -------------------- 加载 JSON 数据 --------------------
def load_json_samples(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)   # 返回列表，每个元素是 [label, img_path, ann_path]
    # 过滤掉不属于我们类别的样本（实际应该全部属于）
    samples = [item for item in data if item[0] in EMOTIONS]
    return samples

print("加载数据...")
train_samples = load_json_samples(TRAIN_JSON)
val_samples   = load_json_samples(VAL_JSON)

print(f"训练样本数: {len(train_samples)}, 验证样本数: {len(val_samples)}")

train_dataset = EmoSetDataset(train_samples, DATA_ROOT, train_transform)
val_dataset   = EmoSetDataset(val_samples,   DATA_ROOT, val_transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=4, pin_memory=True)
val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=4, pin_memory=True)

# -------------------- 模型定义 --------------------
def create_model():
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, NUM_CLASSES)
    return model

model = create_model().to(DEVICE)

if torch.cuda.device_count() > 1:
    print(f"使用 {torch.cuda.device_count()} 块 GPU")
    model = nn.DataParallel(model)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

# -------------------- 训练函数 --------------------
def train_one_epoch(loader, model, optimizer, criterion):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    pbar = tqdm(loader, desc="Training")
    for images, labels in pbar:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        pbar.set_postfix({'Loss': loss.item(), 'Acc': correct/total})

    epoch_loss = total_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc

def evaluate(loader, model, criterion):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Validating"):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return total_loss / total, correct / total

# -------------------- 训练循环 --------------------
best_acc = 0.0
for epoch in range(1, EPOCHS + 1):
    print(f"\nEpoch {epoch}/{EPOCHS}")
    train_loss, train_acc = train_one_epoch(train_loader, model, optimizer, criterion)
    val_loss, val_acc = evaluate(val_loader, model, criterion)
    scheduler.step()

    print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
    print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

    if val_acc > best_acc:
        best_acc = val_acc
        # 保存时去除 DataParallel 前缀
        state_dict = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
        torch.save(state_dict, MODEL_SAVE_PATH)
        print(f"✅ 模型已保存，验证准确率 {best_acc:.4f}")

print(f"\n🎉 训练完成！最佳验证准确率: {best_acc:.4f}")
