# ============================================================
# test_only_script.py
# 仅测试功能 - 适配原训练脚本的数据集/模型/预处理
# ============================================================
import os
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import torch.nn as nn
from spikingjelly.activation_based import neuron, functional

# -------------------------- 导入自定义量化层（和训练脚本一致） --------------------------
# 确保qat_layer.py在当前目录
from qat_layer import Quan_Linear

# -------------------------- 超参数（和训练脚本完全一致） --------------------------
BATCH_SIZE = 128
NUM_CLASSES = 5  # 4类有效数据 + 1类empty类
IMG_SIZE = (48, 48)
DATA_ROOT = "/Users/bytedance/Desktop/Davis/suit_dataset"  # 你的数据集根目录
VALID_CLASSES = ['fangpian', 'heitao', 'hongtao', 'meihua']  # 4个有效类别
EMPTY_CLASS_ID = 4  # empty类的标签ID
MODEL_WEIGHT_PATH = "../SNN/conv+fc_5_class_best.pth"  # 训练好的最佳模型权重
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# -------------------------- 预处理（仅保留测试集，和训练脚本一致） --------------------------
test_transform = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                         std=[0.229, 0.224, 0.225])
])

# -------------------------- 数据集类（完全复用训练脚本） --------------------------
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
        except Exception as e:
            print(f"读取图片失败 {image_path}: {e}")
            image = Image.new('RGB', IMG_SIZE, (0, 0, 0))
        
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label

# -------------------------- 数据收集（仅保留测试集逻辑） --------------------------
def collect_test_data(root_dir):
    """
    从test目录收集数据（复用训练脚本的目录结构逻辑）
    目录结构：
    root_dir/
        test/
            fangpian/
            heitao/
            hongtao/
            meihua/
            empty/
    """
    test_data = []
    test_labels = []
    
    # 遍历test目录
    test_path = os.path.join(root_dir, 'test')
    if not os.path.isdir(test_path):
        raise ValueError(f"未找到test目录，请检查数据集路径：{test_path}")
    
    # 收集4个有效类别
    for class_idx, class_name in enumerate(VALID_CLASSES):
        class_path = os.path.join(test_path, class_name)
        if not os.path.isdir(class_path):
            print(f"警告：未找到test集的{class_name}目录，跳过")
            continue
        
        # 收集该类别下所有图片
        for root, dirs, files in os.walk(class_path):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg')) and not file.startswith('.DS_Store'):
                    img_path = os.path.join(root, file)
                    test_data.append(img_path)
                    test_labels.append(class_idx)
    
    # 收集empty类
    empty_path = os.path.join(test_path, 'empty')
    if os.path.isdir(empty_path):
        for root, dirs, files in os.walk(empty_path):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg')) and not file.startswith('.DS_Store'):
                    img_path = os.path.join(root, file)
                    test_data.append(img_path)
                    test_labels.append(EMPTY_CLASS_ID)
    else:
        print(f"警告：未找到test集的empty目录，将从有效类别中抽取部分作为empty")
        # 兼容原有逻辑：从有效类别图片中抽取前N张作为empty
        # 提取所有有效类别图片
        all_valid_data = [p for p, l in zip(test_data, test_labels) if l < 4]
        all_valid_labels = [l for l in test_labels if l < 4]
        
        new_test_data = []
        new_test_labels = []
        for class_idx in range(4):
            # 提取该类所有图片
            class_img_paths = [p for p, l in zip(all_valid_data, all_valid_labels) if l == class_idx]
            if len(class_img_paths) == 0:
                continue
            
            # 划分有效和empty（最后20张为有效，其余为empty）
            valid_start = max(0, len(class_img_paths) - 20)
            valid_imgs = class_img_paths[valid_start:]
            empty_imgs = class_img_paths[:valid_start]
            
            # 添加有效类
            new_test_data.extend(valid_imgs)
            new_test_labels.extend([class_idx] * len(valid_imgs))
            # 添加empty类
            new_test_data.extend(empty_imgs)
            new_test_labels.extend([EMPTY_CLASS_ID] * len(empty_imgs))
        
        # 更新测试数据
        test_data = new_test_data
        test_labels = new_test_labels
    
    # 统计测试集分布
    test_count = np.bincount(test_labels, minlength=NUM_CLASSES)
    print(f"测试集分布（0-3：有效类，4：empty）：{test_count}")
    print(f"测试集总样本数：{len(test_data)}")
    
    return test_data, test_labels

# -------------------------- 模型定义（完全复用训练脚本） --------------------------
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
        self.fc1 = nn.Linear(in_features=128 * 6 * 6, out_features=512, bias=False)
        self.fc1 = Quan_Linear(w_bits=8, in_features=128 * 6 * 6, out_features=512, bias=False)

        self.lif1 = neuron.LIFNode()
        # self.fc2 = nn.Linear(in_features=512, out_features=128, bias=False)
        self.fc2 = Quan_Linear(w_bits=8, in_features=512, out_features=128, bias=False)

        self.lif2 = neuron.LIFNode()
        # self.fc3 = nn.Linear(in_features=128, out_features=num_classes, bias=False)
        self.fc3 = Quan_Linear(w_bits=8, in_features=128, out_features=5, bias=False)
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

# -------------------------- 详细测试函数（增强版） --------------------------
@torch.no_grad()
def detailed_test(model, test_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    # 每类准确率统计
    class_correct = [0] * NUM_CLASSES
    class_total = [0] * NUM_CLASSES
    class_names = VALID_CLASSES + ['empty']  # 类别名称映射
    
    # 测试批次进度条
    pbar = tqdm(enumerate(test_loader), total=len(test_loader), desc=f"Testing on {device}")
    
    for batch_idx, (inputs, labels) in pbar:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        running_loss += loss.item()
        
        # 计算预测结果
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        # 统计每类结果
        for label, pred in zip(labels, predicted):
            label_idx = label.item()
            pred_idx = pred.item()
            class_total[label_idx] += 1
            if label_idx == pred_idx:
                class_correct[label_idx] += 1
        
        # 更新进度条信息
        batch_loss = running_loss / (batch_idx + 1)
        batch_acc = correct / total
        pbar.set_postfix({
            'batch_loss': f'{batch_loss:.4f}',
            'batch_acc': f'{batch_acc:.4f}'
        })
    
    # 关闭进度条
    pbar.close()
    
    # 计算总体指标
    test_loss = running_loss / len(test_loader)
    test_accuracy = correct / total
    
    # 打印详细结果
    print("\n" + "="*70)
    print(f"📊 测试集最终结果（设备：{device}）")
    print("="*70)
    print(f"总体损失: {test_loss:.4f}")
    print(f"总体准确率: {test_accuracy:.4f} ({correct}/{total})")
    print("\n📈 每类准确率详情：")
    for i in range(NUM_CLASSES):
        if class_total[i] == 0:
            print(f"  • {class_names[i]}: 无测试样本")
        else:
            class_acc = class_correct[i] / class_total[i]
            print(f"  • {class_names[i]}: {class_acc:.4f} ({class_correct[i]}/{class_total[i]})")
    print("="*70)
    
    return test_loss, test_accuracy

# -------------------------- 主函数（仅测试） --------------------------
def main():
    # 1. 加载测试集
    print("===== 加载测试集 =====")
    test_data, test_labels = collect_test_data(DATA_ROOT)
    test_dataset = CustomDataset(test_data, test_labels, transform=test_transform)
    
    # 数据加载器（复用训练脚本的num_workers=0，兼容macOS）
    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False, 
        num_workers=0, pin_memory=True
    )
    
    # 2. 加载模型
    print("\n===== 加载模型 =====")
    model = ConvFCNet(num_classes=NUM_CLASSES, time_steps=3)
    model = model.to(DEVICE)
    
    # 加载权重（兼容checkpoint格式）
    if not os.path.exists(MODEL_WEIGHT_PATH):
        raise FileNotFoundError(f"模型权重文件不存在：{MODEL_WEIGHT_PATH}")
    
    state_dict = torch.load(MODEL_WEIGHT_PATH, map_location=DEVICE)
    # 兼容带optimizer的checkpoint
    if "model_state_dict" in state_dict:
        state_dict = state_dict["model_state_dict"]
    model.load_state_dict(state_dict, strict=False)
    print(f"✅ 成功加载模型权重：{MODEL_WEIGHT_PATH}")
    
    # 3. 定义损失函数（和训练一致）
    criterion = nn.CrossEntropyLoss()
    
    # 4. 执行测试
    print("\n===== 开始测试 =====")
    detailed_test(model, test_loader, criterion, DEVICE)
    
    print("\n🎉 测试完成！")

if __name__ == "__main__":
    # 设置随机种子（保证结果可复现）
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    
    main()