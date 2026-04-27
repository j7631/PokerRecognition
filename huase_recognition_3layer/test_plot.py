import os
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib import font_manager
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse
from torchvision import transforms
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

# ========== 解决中文显示问题（兼容所有系统） ==========
def setup_chinese_font():
    """自动适配系统设置中文字体，避免乱码/警告"""
    try:
        # 优先查找系统中文字体路径
        font_paths = [
            'C:/Windows/Fonts/simhei.ttf',          # Windows
            '/usr/share/fonts/truetype/wqy/wqy-microhei.ttf',  # Linux
            '/Library/Fonts/Arial Unicode MS.ttf',  # Mac
            '/System/Library/Fonts/PingFang.ttc'    # Mac
        ]
        
        usable_font = None
        for path in font_paths:
            if os.path.exists(path):
                usable_font = path
                break
        
        if usable_font:
            font_prop = font_manager.FontProperties(fname=usable_font)
            plt.rcParams['font.family'] = font_prop.get_name()
        else:
            plt.rcParams['font.family'] = 'DejaVu Sans'
            print("⚠️ 无中文字体，将使用英文显示类别")
    except Exception:
        plt.rcParams['font.family'] = 'sans-serif'
        print("⚠️ 字体设置失败，使用英文显示")
    
    plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示

# 初始化字体
setup_chinese_font()

# ========== 导入原代码核心配置和类 ==========
# 从你的训练代码中复制关键配置
BATCH_SIZE = 128
NUM_CLASSES = 5
IMG_SIZE = (48, 48)
DATA_ROOT = "/Users/bytedance/Desktop/Davis/suit_dataset"  # 和训练代码一致
VALID_CLASSES = ['fangpian', 'heitao', 'hongtao', 'meihua']
EMPTY_CLASS_ID = 4
NUM_WORKERS = 0  # macOS必须设为0

# 类别名称（中英文自适应）
try:
    plt.text(0.5, 0.5, '测试', fontsize=12)
    CLASS_NAMES = ['方片', '黑桃', '红桃', '梅花']
    PLOT_TITLE = 'ConvFCNet测试集四类准确率（不含empty）'
    X_LABEL = '花色类别'
    Y_LABEL = '准确率 (%)'
except:
    CLASS_NAMES = ['Fangpian', 'Heitao', 'Hongtao', 'Meihua']
    PLOT_TITLE = 'ConvFCNet Test Accuracy (4 Classes, No Empty)'
    X_LABEL = 'Card Suit'
    Y_LABEL = 'Accuracy (%)'

CLASS_IDXS = [0, 1, 2, 3]  # 只计算前四类

# 复制原代码的关键类和函数
class AddGaussianNoise(object):
    def __init__(self, mean=0.0, std=0.01):
        self.mean = mean
        self.std = std
    def __call__(self, tensor):
        return tensor + self.std * torch.randn_like(tensor)
    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)

# 测试集预处理（和训练代码完全一致）
test_transform = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                         std=[0.229, 0.224, 0.225])
])

class CustomDataset(Dataset):
    def __init__(self, data, labels, transform=None):
        self.data = data
        self.labels = labels
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

def collect_data_from_train_test(root_dir):
    """仅收集测试集数据（简化版）"""
    test_data = []
    test_labels = []
    
    # 只处理test目录
    split = 'test'
    split_path = os.path.join(root_dir, split)
    if not os.path.isdir(split_path):
        raise ValueError(f"未找到test目录：{split_path}")
    
    # 收集4个有效类别
    for class_idx, class_name in enumerate(VALID_CLASSES):
        class_path = os.path.join(split_path, class_name)
        if not os.path.isdir(class_path):
            print(f"警告：未找到test集的{class_name}目录，跳过")
            continue
        for root, dirs, files in os.walk(class_path):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg')) and not file.startswith('.DS_Store'):
                    img_path = os.path.join(root, file)
                    test_data.append(img_path)
                    test_labels.append(class_idx)
    
    # 收集empty类（但后续统计会跳过）
    empty_path = os.path.join(split_path, 'empty')
    if os.path.isdir(empty_path):
        for root, dirs, files in os.walk(empty_path):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg')) and not file.startswith('.DS_Store'):
                    img_path = os.path.join(root, file)
                    test_data.append(img_path)
                    test_labels.append(EMPTY_CLASS_ID)
    else:
        print(f"警告：未找到test集的empty目录，从有效类别抽取部分作为empty")
        # 兼容原有逻辑
        new_test_data = []
        new_test_labels = []
        for class_idx in range(4):
            class_img_paths = [p for p, l in zip(test_data, test_labels) if l == class_idx]
            if len(class_img_paths) == 0:
                continue
            valid_start = max(0, len(class_img_paths) - 20)
            valid_imgs = class_img_paths[valid_start:]
            empty_imgs = class_img_paths[:valid_start]
            new_test_data.extend(valid_imgs)
            new_test_labels.extend([class_idx] * len(valid_imgs))
            new_test_data.extend(empty_imgs)
            new_test_labels.extend([EMPTY_CLASS_ID] * len(empty_imgs))
        test_data = new_test_data
        test_labels = new_test_labels
    
    print(f"测试集加载完成：共{len(test_data)}张图片")
    return test_data, test_labels

# 复制原代码的模型定义
from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image
from spikingjelly.activation_based import neuron, functional
from qat_layer import Quan_Linear  # 确保qat_layer.py在当前目录
import torch.nn as nn

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

# ========== 核心功能函数 ==========
def load_model(model_path, device):
    """加载ConvFCNet模型"""
    model = ConvFCNet(num_classes=NUM_CLASSES, time_steps=3)
    checkpoint = torch.load(model_path, map_location=device)
    
    # 兼容两种保存方式
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.to(device)
    model.eval()
    return model

def calculate_class_accuracy(model, test_loader, device, class_idxs, class_names):
    """计算前四类准确率（跳过empty）"""
    class_correct = np.zeros(len(class_idxs))
    class_total = np.zeros(len(class_idxs))

    with torch.no_grad():
        pbar = tqdm(test_loader, desc="Calculating Class Accuracy")
        for inputs, labels in pbar:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            # 模型前向传播
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)

            # 只统计前四类
            for label, pred in zip(labels, predicted):
                lb = label.item()
                if lb not in class_idxs:
                    continue
                cls_idx = class_idxs.index(lb)
                class_total[cls_idx] += 1
                if pred == label:
                    class_correct[cls_idx] += 1

    # 计算准确率（处理除零）
    class_accuracy = np.divide(
        class_correct, class_total,
        out=np.zeros_like(class_correct),
        where=class_total != 0
    ) * 100

    # 打印结果
    print("\n===== 四类准确率（不含empty）=====")
    for name, acc, cor, total in zip(class_names, class_accuracy, class_correct, class_total):
        print(f"{name}: {acc:.2f}% ({int(cor)}/{int(total)})")

    return class_accuracy

def plot_class_accuracy(class_accuracy, class_names, save_path):
    """绘制柱状图"""
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']
    fig, ax = plt.subplots(figsize=(9, 6), dpi=100)

    # 绘制柱状图
    bars = ax.bar(class_names, class_accuracy, color=colors, edgecolor='white', linewidth=1.2)

    # 添加数值标签
    for bar, acc in zip(bars, class_accuracy):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + 1.5,
            f'{acc:.2f}%',
            ha='center', va='bottom',
            fontsize=11, fontweight='bold'
        )

    # 图表样式
    ax.set_title(PLOT_TITLE, fontsize=16, pad=20, fontweight='bold')
    ax.set_xlabel(X_LABEL, fontsize=14, labelpad=10)
    ax.set_ylabel(Y_LABEL, fontsize=14, labelpad=10)
    ax.set_ylim(0, 105)
    ax.yaxis.grid(True, linestyle='--', alpha=0.3, color='gray')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_axisbelow(True)

    # 保存并显示
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n✅ 图表已保存至：{save_path}")
    plt.show()

def main(args):
    # 1. 设备设置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 2. 加载测试集
    print("\n===== 加载测试数据集 =====")
    test_data, test_labels = collect_data_from_train_test(DATA_ROOT)
    test_dataset = CustomDataset(test_data, test_labels, transform=test_transform)
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )
    
    # 3. 加载模型
    print("\n===== 加载训练好的模型 =====")
    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"模型文件不存在: {args.model_path}")
    model = load_model(args.model_path, device)
    
    # 4. 计算准确率
    class_accuracy = calculate_class_accuracy(
        model, test_loader, device, CLASS_IDXS, CLASS_NAMES
    )
    
    # 5. 绘图
    plot_class_accuracy(class_accuracy, CLASS_NAMES, args.save_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='绘制ConvFCNet四类准确率')
    parser.add_argument('--model_path', type=str, required=True, 
                        help='模型路径 (如 conv+fc_5_class_best.pth)')
    parser.add_argument('--save_path', type=str, default='convfc_class_acc_4cls.png',
                        help='图表保存路径')
    args = parser.parse_args()
    main(args)

    #  /usr/bin/python3 test_plot.py --model_path /Users/bytedance/Desktop/bishe/2_huase_augment/conv+fc_5_class_best.pth  --save_path /Users/bytedance/Desktop/bishe/2_huase_augmentclass_accuracy.png