import os
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
from sklearn.model_selection import train_test_split
import torch.optim as optim
import torch.nn as nn
from spikingjelly.activation_based import neuron, functional, surrogate, layer
# PyTorch QAT FX
from torch.ao.quantization import get_default_qat_qconfig
from torch.ao.quantization.quantize_fx import prepare_qat_fx, convert_fx
from torch.ao.quantization.qconfig_mapping import QConfigMapping
from spikingjelly.activation_based import neuron, layer, functional

from tqdm import tqdm


# 自定义量化层（确保qat_layer.py在当前目录）
from qat_layer import Quan_Linear

# -------------------------- 解决Lambda序列化问题：定义可序列化的噪声函数 --------------------------
class AddGaussianNoise(object):
    """可序列化的高斯噪声添加类（替代lambda函数）"""
    def __init__(self, mean=0.0, std=0.01):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        return tensor + self.std * torch.randn_like(tensor)

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)

# -------------------------- 超参数定义 --------------------------
BATCH_SIZE = 128
NUM_CLASSES = 5  # 4类有效数据 + 1类empty类
NUM_EPOCHS = 80
LEARNING_RATE = 0.0001
IMG_SIZE = (48, 48)  # 与原有预处理一致
AUGMENT_MULTIPLIER = 20  # 数据增强倍数（可调整）
MIN_SAMPLES_THRESHOLD = 10  # 类别最小样本阈值
DATA_ROOT = "/Users/bytedance/Desktop/Davis/suit_dataset"  # 你的数据集根目录（已分train/test）
VALID_CLASSES = ['fangpian', 'heitao', 'hongtao', 'meihua']  # 4个有效类别
EMPTY_CLASS_ID = 4  # empty类的标签ID
NUM_WORKERS = 0
MAX_SEQ_LEN = 10
# -------------------------- 增强版数据预处理与增强 --------------------------
# 训练集：极致数据增强（覆盖几何、颜色、像素级变换）
# -------------------------- 增强版数据预处理与增强 --------------------------
# 训练集：极致数据增强（覆盖几何、颜色、像素级变换）
train_transform = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    # 几何变换（作用于PIL Image）
    transforms.RandomRotation(degrees=(-30, 30)),  # 扩大旋转角度范围
    transforms.RandomHorizontalFlip(p=0.5),  # 随机水平翻转
    transforms.RandomVerticalFlip(p=0.2),  # 增加随机垂直翻转
    transforms.RandomAffine(
        degrees=0, 
        translate=(0.2, 0.2),  # 随机平移
        scale=(0.8, 1.2),      # 随机缩放
        shear=(-15, 15)        # 随机剪切
    ),
    # 颜色变换（作用于PIL Image）- 修正hue范围
    transforms.ColorJitter(
        brightness=(0.4, 1.6),  # 亮度范围扩大
        contrast=(0.4, 1.6),    # 增加对比度变换
        # saturation=(0.4, 1.6),  # 增加饱和度变换
        # hue=(-0.05, 0.05)       # 大幅缩小hue范围，避免uint8溢出
    ),
    # 像素级变换（PIL Image）
    transforms.RandomGrayscale(p=0.1),  # 随机灰度化
    transforms.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 2.0)),  # 随机高斯模糊
    # 先转Tensor，再做需要Tensor的变换
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                         std=[0.229, 0.224, 0.225]),
    # 随机擦除（必须在ToTensor之后，作用于Tensor）
    transforms.RandomErasing(p=0.2, scale=(0.02, 0.2), ratio=(0.3, 3.3)),
    # 随机噪声（用自定义可序列化类替代lambda）
    AddGaussianNoise(mean=0.0, std=0.01)
])


# 测试集：仅基础预处理（无增强，保证评估准确性）
test_transform = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                         std=[0.229, 0.224, 0.225])
])

# -------------------------- 数据集类（适配train/test目录结构） --------------------------
class CustomDataset(Dataset):
    def __init__(self, data, labels, transform=None):
        self.data = data  # 图片路径列表
        self.labels = labels  # 标签列表
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_path = self.data[idx]
        try:
            image = Image.open(image_path).convert('RGB')
            # 移除手动resize，交给transforms.Resize处理
            # 避免重复resize导致变换顺序错误
        except Exception as e:
            print(f"读取图片失败 {image_path}: {e}")
            # 返回空图片兜底
            image = Image.new('RGB', IMG_SIZE, (0, 0, 0))
        
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label

# -------------------------- 数据收集（适配已分好的train/test目录） --------------------------
def collect_data_from_train_test(root_dir):
    """
    从已分好的train/test目录收集数据
    目录结构：
    root_dir/
        train/
            fangpian/
            heitao/
            hongtao/
            meihua/
            empty/  # 新增empty类目录（建议你按此结构整理）
        test/
            fangpian/
            heitao/
            hongtao/
            meihua/
            empty/
    """
    # 初始化训练/测试数据存储
    train_data = []
    train_labels = []
    test_data = []
    test_labels = []
    
    # 遍历train和test目录
    for split in ['train', 'test']:
        split_path = os.path.join(root_dir, split)
        if not os.path.isdir(split_path):
            raise ValueError(f"未找到{split}目录，请检查数据集路径：{split_path}")
        
        # 收集4个有效类别
        for class_idx, class_name in enumerate(VALID_CLASSES):
            class_path = os.path.join(split_path, class_name)
            if not os.path.isdir(class_path):
                print(f"警告：未找到{split}集的{class_name}目录，跳过")
                continue
            
            # 收集该类别下所有图片
            for root, dirs, files in os.walk(class_path):
                for file in files:
                    if file.lower().endswith(('.png', '.jpg', '.jpeg')) and not file.startswith('.DS_Store'):
                        img_path = os.path.join(root, file)
                        if split == 'train':
                            train_data.append(img_path)
                            train_labels.append(class_idx)
                        else:
                            test_data.append(img_path)
                            test_labels.append(class_idx)
        
        # 收集empty类
        empty_path = os.path.join(split_path, 'empty')
        if os.path.isdir(empty_path):
            for root, dirs, files in os.walk(empty_path):
                for file in files:
                    if file.lower().endswith(('.png', '.jpg', '.jpeg')) and not file.startswith('.DS_Store'):
                        img_path = os.path.join(root, file)
                        if split == 'train':
                            train_data.append(img_path)
                            train_labels.append(EMPTY_CLASS_ID)
                        else:
                            test_data.append(img_path)
                            test_labels.append(EMPTY_CLASS_ID)
        else:
            print(f"警告：未找到{split}集的empty目录，将从有效类别中抽取部分作为empty")
            # 兼容原有逻辑：从有效类别图片中抽取前N张作为empty
            split_all_data = train_data if split == 'train' else test_data
            split_all_labels = train_labels if split == 'train' else test_labels
            
            # 重新划分empty类（最后20张为有效，其余为empty）
            new_split_data = []
            new_split_labels = []
            for class_idx in range(4):
                # 提取该类所有图片
                class_img_paths = [p for p, l in zip(split_all_data, split_all_labels) if l == class_idx]
                if len(class_img_paths) == 0:
                    continue
                
                # 划分有效和empty
                valid_start = max(0, len(class_img_paths) - 20)
                valid_imgs = class_img_paths[valid_start:]
                empty_imgs = class_img_paths[:valid_start]
                
                # 添加有效类
                new_split_data.extend(valid_imgs)
                new_split_labels.extend([class_idx] * len(valid_imgs))
                # 添加empty类
                new_split_data.extend(empty_imgs)
                new_split_labels.extend([EMPTY_CLASS_ID] * len(empty_imgs))
            
            # 更新数据
            if split == 'train':
                train_data = new_split_data
                train_labels = new_split_labels
            else:
                test_data = new_split_data
                test_labels = new_split_labels
    
    # 统计原始数据分布
    train_count = np.bincount(train_labels, minlength=NUM_CLASSES)
    test_count = np.bincount(test_labels, minlength=NUM_CLASSES)
    print(f"原始训练集分布（0-3：有效类，4：empty）：{train_count}")
    print(f"原始测试集分布（0-3：有效类，4：empty）：{test_count}")
    
    return train_data, train_labels, test_data, test_labels

def balance_classes(data, labels):
    """平衡类别样本数量"""
    class_counts = np.bincount(labels, minlength=NUM_CLASSES)
    print(f"平衡前样本数：{class_counts}")
    
    # 检查样本阈值
    for cls, count in enumerate(class_counts):
        if count < MIN_SAMPLES_THRESHOLD:
            raise ValueError(f"类别{cls}样本数不足（{count}张），请补充数据或降低MIN_SAMPLES_THRESHOLD")
    
    # 以最多样本数为基准（增强少样本类）
    max_count = max(class_counts)
    print(f"平衡目标：每个类别增强至{max_count}张")
    
    balanced_data = []
    balanced_labels = []
    for cls in range(NUM_CLASSES):
        # 提取该类所有样本
        cls_data = [p for p, l in zip(data, labels) if l == cls]
        cls_len = len(cls_data)
        
        if cls_len == 0:
            continue
        
        # 重复采样至max_count（用于后续增强）
        if cls_len < max_count:
            # 随机重复采样（有放回）
            repeat_times = max_count // cls_len
            remainder = max_count % cls_len
            balanced_cls_data = cls_data * repeat_times + list(np.random.choice(cls_data, remainder, replace=True))
        else:
            # 随机采样至max_count（无放回）
            balanced_cls_data = list(np.random.choice(cls_data, max_count, replace=False))
        
        balanced_data.extend(balanced_cls_data)
        balanced_labels.extend([cls] * max_count)
    
    return balanced_data, balanced_labels

def augment_train_data_enhanced(train_data, train_labels, augment_multiplier):
    """增强版数据增强：按类别分层增强，保证多样性"""
    # 按类别分组
    class_groups = {cls: [] for cls in range(NUM_CLASSES)}
    for path, lbl in zip(train_data, train_labels):
        class_groups[lbl].append(path)
    
    augmented_data = []
    augmented_labels = []
    # 每个类别目标数量 = 原始数量 * 增强倍数
    for cls, paths in class_groups.items():
        original_count = len(paths)
        target_count = original_count * augment_multiplier
        print(f"类别{cls}：原始{original_count}张 → 增强至{target_count}张")
        
        # 先添加原始样本
        augmented_data.extend(paths)
        augmented_labels.extend([cls] * original_count)
        
        # 计算需要增强的数量
        need_augment = target_count - original_count
        if need_augment <= 0:
            continue
        
        # 分层增强：多次随机采样+不同增强策略（通过transform实现）
        augment_paths = []
        while len(augment_paths) < need_augment:
            # 每次采样不超过原始数量，保证多样性
            batch_size = min(original_count, need_augment - len(augment_paths))
            augment_batch = np.random.choice(paths, batch_size, replace=True)
            augment_paths.extend(augment_batch)
        
        augmented_data.extend(augment_paths)
        augmented_labels.extend([cls] * need_augment)
    
    # 打乱增强后的数据
    shuffle_idx = np.random.permutation(len(augmented_data))
    augmented_data = [augmented_data[i] for i in shuffle_idx]
    augmented_labels = [augmented_labels[i] for i in shuffle_idx]
    
    return augmented_data, augmented_labels

# -------------------------- 数据加载流程 --------------------------
def load_balanced_augmented_data(root_dir):
    """加载平衡+增强后的数据集"""
    # 1. 从train/test目录收集原始数据
    print("===== 收集train/test数据 =====")
    train_data, train_labels, test_data, test_labels = collect_data_from_train_test(root_dir)
    
    # 2. 平衡训练集（测试集不平衡，保持原始分布）
    print("===== 平衡训练集类别 =====")
    train_data_balanced, train_labels_balanced = balance_classes(train_data, train_labels)
    
    # 3. 训练集极致数据增强
    print("===== 训练集极致数据增强 =====")
    train_data_aug, train_labels_aug = augment_train_data_enhanced(
        train_data_balanced, train_labels_balanced, AUGMENT_MULTIPLIER
    )
    
    # 验证增强后的分布
    train_aug_count = np.bincount(train_labels_aug, minlength=NUM_CLASSES)
    test_count = np.bincount(test_labels, minlength=NUM_CLASSES)
    print(f"增强后训练集规模：{len(train_data_aug)}张，分布：{train_aug_count}")
    print(f"测试集规模：{len(test_data)}张，分布：{test_count}")
    
    # 4. 创建Dataset和DataLoader
    train_dataset = CustomDataset(train_data_aug, train_labels_aug, transform=train_transform)
    test_dataset = CustomDataset(test_data, test_labels, transform=test_transform)
    
    # 关键修复：macOS强制使用num_workers=0，避免多进程序列化问题
    num_workers = 0  # 统一设置为0，兼容macOS/Windows/Linux
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, 
        num_workers=num_workers, pin_memory=True, drop_last=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False, 
        num_workers=num_workers, pin_memory=True
    )
    
    return train_loader, test_loader

# -------------------------- 模型定义（保留原有） --------------------------
class ConvFCNet(nn.Module):
    def __init__(self, num_classes, time_steps=3):
        super(ConvFCNet, self).__init__()
        self.time_steps = time_steps

        self.conv_layer = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.fc1 = Quan_Linear(w_bits=8, in_features=128 * 6 * 6, out_features=512, bias=False)
        self.lif1 = neuron.LIFNode()
        self.fc2 = Quan_Linear(w_bits=8, in_features=512, out_features=128, bias=False)
        self.lif2 = neuron.LIFNode()
        self.fc3 = Quan_Linear(w_bits=8, in_features=128, out_features=num_classes, bias=False)    
        self.lif3 = neuron.LIFNode()

    def forward(self, x):
        x = self.conv_layer(x)
        x = torch.flatten(x, 1)
        functional.reset_net(self)
        out_spk = 0.0
        for t in range(self.time_steps):
            cur = self.fc1(x)
            spk1 = self.lif1(cur)
            cur2 = self.fc2(spk1)
            spk2 = self.lif2(cur2)
            cur3 = self.fc3(spk2)
            spk3 = self.lif3(cur3)
            out_spk += spk3
        out_spk = out_spk / self.time_steps
        return out_spk

# -------------------------- 训练/测试函数（保留原有） --------------------------
def train(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    # 新增：创建批次进度条，描述为"Training Batches"
    pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Training")
    for batch_idx, (inputs, labels) in pbar:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        # 新增：更新进度条显示的实时信息
        batch_loss = running_loss / (batch_idx + 1)
        batch_acc = correct / total
        pbar.set_postfix({
            'batch_loss': f'{batch_loss:.4f}',
            'batch_acc': f'{batch_acc:.4f}'
        })
    
    # 关闭进度条
    pbar.close()
    
    train_loss = running_loss / len(train_loader)
    train_accuracy = correct / total
    return train_loss, train_accuracy

def test(model, test_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    # 新增：创建测试批次进度条
    pbar = tqdm(enumerate(test_loader), total=len(test_loader), desc=f"Testing")
    with torch.no_grad():
        for batch_idx, (inputs, labels) in pbar:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # 新增：更新测试进度条信息
            batch_loss = running_loss / (batch_idx + 1)
            batch_acc = correct / total
            pbar.set_postfix({
                'batch_loss': f'{batch_loss:.4f}',
                'batch_acc': f'{batch_acc:.4f}'
            })
    
    # 关闭进度条
    pbar.close()
    
    test_loss = running_loss / len(test_loader)
    test_accuracy = correct / total
    return test_loss, test_accuracy

# -------------------------- Checkpoint函数（保留原有） --------------------------
def save_checkpoint(state, filename='checkpoint.pth'):
    torch.save(state, filename)

def load_checkpoint(model, optimizer, filename='checkpoint.pth'):
    if os.path.isfile(filename):
        print(f"=> loading checkpoint '{filename}'")
        checkpoint = torch.load(filename)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_test_accuracy = checkpoint['best_test_accuracy']
        print(f"=> loaded checkpoint '{filename}' (epoch {checkpoint['epoch']})")
        return start_epoch, best_test_accuracy
    else:
        print(f"=> no checkpoint found at '{filename}'")
        return 0, 0.0

# -------------------------- 主函数（适配新的数据集路径） --------------------------
def main(args):
    # 1. 加载平衡+增强后的数据集
    print("===== 开始加载数据集 =====")
    train_loader, test_loader = load_balanced_augmented_data(DATA_ROOT)

    # 2. 模型、设备、损失函数、优化器设置
    model = ConvFCNet(num_classes=NUM_CLASSES, time_steps=3)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # 3. 加载checkpoint（如需续训）
    start_epoch = 0
    best_test_accuracy = 0.0
    checkpoint_path = args.checkpoint
    if args.resume and checkpoint_path:
        start_epoch, best_test_accuracy = load_checkpoint(model, optimizer, filename=checkpoint_path)

    # 4. 训练循环 - 新增：epoch总进度条
    print("===== 开始训练 =====")
    # 创建epoch级别的进度条，覆盖所有训练轮次
    epoch_pbar = tqdm(range(start_epoch, NUM_EPOCHS), total=NUM_EPOCHS, desc="Total Training Epochs")
    for epoch in epoch_pbar:
        # 训练
        train_loss, train_accuracy = train(model, train_loader, criterion, optimizer, device)
        # 测试
        test_loss, test_accuracy = test(model, test_loader, criterion, device)
        
        # 更新epoch进度条的信息
        epoch_pbar.set_postfix({
            'train_loss': f'{train_loss:.4f}',
            'train_acc': f'{train_accuracy:.4f}',
            'test_loss': f'{test_loss:.4f}',
            'test_acc': f'{test_accuracy:.4f}',
            'best_test_acc': f'{best_test_accuracy:.4f}'
        })
        
        # 打印详细日志（可选，保留原有打印）
        print(f'\nEpoch [{epoch+1}/{NUM_EPOCHS}], '
              f'Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, '
              f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}')

        # 保存最佳模型
        if test_accuracy > best_test_accuracy:
            best_test_accuracy = test_accuracy
            torch.save(model.state_dict(), 'conv+fc_5_class_best.pth')
            print(f"更新最佳模型，当前最佳测试准确率：{best_test_accuracy:.4f}")

        # 保存checkpoint
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_test_accuracy': best_test_accuracy
        }
        save_checkpoint(checkpoint, filename=checkpoint_path)
    
    # 关闭epoch进度条
    epoch_pbar.close()

    # 保存最终模型
    torch.save(model.state_dict(), 'conv+fc_class_final.pth')
    print("===== 训练完成 =====")

# 关键修复：macOS多进程需要保护主函数入口
if __name__ == "__main__":
    # 设置随机种子，保证结果可复现
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume', action='store_true', help='Resume training from checkpoint')
    parser.add_argument('--checkpoint', type=str, default='conv+fc_5_class_checkpoint.pth', help='Path to checkpoint file')
    args = parser.parse_args()
    main(args)