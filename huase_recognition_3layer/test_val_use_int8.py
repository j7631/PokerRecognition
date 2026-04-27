# ============================================================
# test_val_use_int8.py
# 修复：权重维度不匹配 + LIF膜电位重置 + 维度校验
# ============================================================

import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
from spikingjelly.activation_based import neuron, functional
from qat_layer import Quan_Linear

# ========================= 1. 配置参数（修改为你的实际路径） =========================
TEST_DATA_DIR = "/Users/bytedance/Desktop/Davis/suit_dataset/test"
MODEL_PATH = "/Users/bytedance/Desktop/bishe/2_huase_augment_copy/conv+fc_5_class_best.pth"
FC1_NPY_PATH = "/Users/bytedance/Desktop/bishe/2_huase_augment_copy/save_int8_qat_compatible/fc1_weight_int8.npy"
FC2_NPY_PATH = "/Users/bytedance/Desktop/bishe/2_huase_augment_copy/save_int8_qat_compatible/fc2_weight_int8.npy"
FC3_NPY_PATH = "/Users/bytedance/Desktop/bishe/2_huase_augment_copy/save_int8_qat_compatible/fc3_weight_int8.npy"

# 模型配置
CLASS_NAMES = ["empty", "fangpian", "heitao", "hongtao", "meihua"]
NUM_CLASSES = len(CLASS_NAMES)
TIME_STEPS = 3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 数据配置
BATCH_SIZE = 32
IMAGE_SIZE = (48, 48)
OUTPUT_TO_TRUE = {
    0: 1,  
    1: 2,  
    2: 3,  
    3: 4,  
    4: 0   
}

# ========================= 2. 数据加载 =========================
class CustomDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []

        for label_idx, class_name in enumerate(CLASS_NAMES):
            class_dir = os.path.join(data_dir, class_name)
            if not os.path.exists(class_dir):
                print(f"⚠️  类别文件夹 {class_dir} 不存在，跳过")
                continue
            for img_name in os.listdir(class_dir):
                if img_name.endswith((".jpg", ".png", ".jpeg")):
                    self.image_paths.append(os.path.join(class_dir, img_name))
                    self.labels.append(label_idx)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        
        img = Image.open(img_path).convert("RGB")
        
        if self.transform:
            img = self.transform(img)
        
        return img, label

# ========================= 3. 模型定义（核心修复） =========================
class FloatPart(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_layer = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
            nn.Conv2d(64,128,3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2,2)
        )


    def forward(self, x):
        x = self.conv_layer(x)
        x = torch.flatten(x, 1)
        # x = self.fc1(x)
        # spk1 = self.lif1(x)
        # return spk1
        return x

class Int8FCPart(nn.Module):
    def __init__(self, fc1_npy, fc2_npy, fc3_npy):
        super().__init__()
        # 1. 加载并校验权重维度
        fc1_weight = np.load(fc1_npy).astype(np.float32)  # 直接转float32（int8转float不影响维度）
        fc2_weight = np.load(fc2_npy).astype(np.float32)  # 直接转float32（int8转float不影响维度）
        fc3_weight = np.load(fc3_npy).astype(np.float32)
        
        # 打印权重维度（方便调试）
        print(f"🔍 fc1权重维度: {fc1_weight.shape} (期望: [512, 4608])")
        print(f"🔍 fc2权重维度: {fc2_weight.shape} (期望: [128, 512])")
        print(f"🔍 fc3权重维度: {fc3_weight.shape} (期望: [{NUM_CLASSES}, 128])")
        
        # 维度校验
        if fc1_weight.shape != (512, 4608):
            raise ValueError(f"fc1权重维度错误！期望(512,4608])，实际{fc1_weight.shape}")
        if fc2_weight.shape != (128, 512):
            raise ValueError(f"fc2权重维度错误！期望(128,512)，实际{fc2_weight.shape}")
        if fc3_weight.shape != (NUM_CLASSES, 128):
            raise ValueError(f"fc3权重维度错误！期望({NUM_CLASSES},128)，实际{fc3_weight.shape}")

        # 2. 定义全连接层并加载权重
        self.fc1 = nn.Linear(in_features=4608, out_features=512, bias=False)
        self.fc2 = nn.Linear(in_features=512, out_features=128, bias=False)
        self.fc3 = nn.Linear(in_features=128, out_features=NUM_CLASSES, bias=False)

        # self.fc1 = Quan_Linear(w_bits=8, in_features=128 * 6 * 6, out_features=512, bias=False)
        # self.fc2 = Quan_Linear(w_bits=8, in_features=512, out_features=128, bias=False)
        # self.fc3 = Quan_Linear(w_bits=8, in_features=128, out_features=5, bias=False)


        # 加载权重（转为torch tensor并移到设备）
        self.fc1.weight.data = torch.from_numpy(fc1_weight).to(DEVICE)
        self.fc2.weight.data = torch.from_numpy(fc2_weight).to(DEVICE)
        self.fc3.weight.data = torch.from_numpy(fc3_weight).to(DEVICE)
        
        # 3. 显式设置LIF参数，避免默认值冲突
        self.lif1 = neuron.LIFNode()
        self.lif2 = neuron.LIFNode()
        self.lif3 = neuron.LIFNode()

    def forward(self, x):
        x = x * 1.0/128.0  # 输入缩放
        x = self.fc1(x)
        x = self.lif1(x)
        x = self.fc2(x)
        x = self.lif2(x)
        x = self.fc3(x)
        x = self.lif3(x)
        return x

class SNNInferModel(nn.Module):
    def __init__(self, model_path, fc1_npy, fc2_npy, fc3_npy, time_steps=TIME_STEPS):
        super().__init__()
        self.time_steps = time_steps
        self.float_part = FloatPart().to(DEVICE)
        self.int8_fc_part = Int8FCPart(fc1_npy, fc2_npy, fc3_npy).to(DEVICE)

        # 加载浮点卷积层权重
        state_dict = torch.load(model_path, map_location=DEVICE)
        if "model_state_dict" in state_dict:
            state_dict = state_dict["model_state_dict"]
        self.float_part.load_state_dict(state_dict, strict=False)
        
        self.float_part.eval()
        self.int8_fc_part.eval()

    def forward(self, x):
        # 核心修复：重置所有LIF神经元的膜电位（包括lif2/lif3）
        functional.reset_net(self.float_part)
        functional.reset_net(self.int8_fc_part)
        
        out_spk = 0.0
        x = x.to(DEVICE)  # 确保输入在正确设备上
        for t in range(self.time_steps):
            spk1 = self.float_part(x)
            out = self.int8_fc_part(spk1)
            out_spk += out
        return out_spk / self.time_steps

# ========================= 4. 测试集推理 =========================
@torch.no_grad()
def test_model(model, test_loader):
    correct = 0
    total = 0
    class_correct = [0] * NUM_CLASSES
    class_total = [0] * NUM_CLASSES

    print("\n🚀 开始测试集推理...")
    for batch_idx, (images, labels) in enumerate(test_loader):
        # 确保数据在正确设备上
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        try:
            # 模型推理
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            

            # 类别映射
            preds_mapped = torch.tensor([OUTPUT_TO_TRUE[p.item()] for p in preds]).to(DEVICE)

            # 统计结果
            total += labels.size(0)
            correct += (preds_mapped == labels).sum().item()

            for label, pred in zip(labels, preds_mapped):
                label_idx = label.item()
                pred_idx = pred.item()
                class_total[label_idx] += 1
                if label_idx == pred_idx:
                    class_correct[label_idx] += 1

            # 打印进度
            if (batch_idx + 1) % 10 == 0:
                batch_acc = (preds_mapped == labels).sum().item() / labels.size(0)
                print(f"批次 {batch_idx+1}/{len(test_loader)} | 批次准确率: {batch_acc:.4f}")
        
        except Exception as e:
            print(f"\n❌ 批次 {batch_idx+1} 推理出错: {str(e)}")
            print(f"   输入图片维度: {images.shape}")
            continue

    # 计算总体准确率
    if total == 0:
        overall_acc = 0.0
    else:
        overall_acc = correct / total
    
    print("\n" + "="*50)
    print(f"📊 测试集总体准确率: {overall_acc:.4f} ({correct}/{total})")
    print("="*50)
    
    # 每类准确率
    print("\n📈 每类准确率详情：")
    for i in range(NUM_CLASSES):
        if class_total[i] == 0:
            print(f"  • {CLASS_NAMES[i]}: 无测试样本")
        else:
            class_acc = class_correct[i] / class_total[i]
            print(f"  • {CLASS_NAMES[i]}: {class_acc:.4f} ({class_correct[i]}/{class_total[i]})")

# ========================= 5. 主程序入口 =========================
if __name__ == '__main__':
    # 数据预处理
    test_transform = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    # 加载测试集
    test_dataset = CustomDataset(TEST_DATA_DIR, transform=test_transform)
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    print(f"✅ 测试集加载完成：共 {len(test_dataset)} 张图片，{len(test_loader)} 个批次")

    # 检查文件是否存在
    for file_path in [MODEL_PATH, FC1_NPY_PATH, FC2_NPY_PATH, FC3_NPY_PATH]:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"文件不存在: {file_path}")
    
    # 加载模型
    model = SNNInferModel(MODEL_PATH, FC1_NPY_PATH, FC2_NPY_PATH, FC3_NPY_PATH, TIME_STEPS).to(DEVICE)
    print("✅ INT8分段模型加载完成（FloatPart + Int8FCPart）")

    # 执行推理（增加异常捕获）
    test_model(model, test_loader)

    print("\n🎉 测试集推理完成！")