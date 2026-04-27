import os
import cv2
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
import torch.nn as nn
from spikingjelly.activation_based import neuron, functional

# ===================== 核心配置（整合INT8推理+事件测试） =====================
# 模型/权重路径
MODEL_PATH = '/Users/bytedance/Desktop/bishe/2_huase_augment/conv+fc_5_class_best.pth'  # 卷积层权重
FC1_WEIGHT_NPY = "/Users/bytedance/Desktop/bishe/2_huase_augment_copy/save_int8_qat_compatible/fc1_weight_int8.npy"
FC2_WEIGHT_NPY = "/Users/bytedance/Desktop/bishe/2_huase_augment_copy/save_int8_qat_compatible/fc2_weight_int8.npy"
FC3_WEIGHT_NPY = "/Users/bytedance/Desktop/bishe/2_huase_augment_copy/save_int8_qat_compatible/fc3_weight_int8.npy"

# 数据/事件配置
INPUT_DIR = "/Users/bytedance/Desktop/dvs_data/test_offline"  # 事件帧目录
REMOVE_FIRST_N = 10  # 跳过前N帧
WHITE_THRESH = 200   # 事件检测阈值
BLUE_THRESH = 200
MIN_GAP_FRAMES = 10  # 事件间隔阈值
MIN_EVENT_FRAMES = 7 # 最小事件帧数量

# 模型参数
NUM_CLASSES = 5
TIME_STEPS = 3
CLASS_NAMES = ['fangpian', 'heitao', 'hongtao', 'meihua', 'empty']
# CLASS_NAMES = ["empty", "♦️", "♠️", "♥️", "♣️"]

IMG_SIZE = (48, 48)

# 设备配置
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ===================== 图片预处理（与训练一致） =====================
test_transform = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                         std=[0.229, 0.224, 0.225])
])

# ===================== 1. 浮点卷积部分（保留INT8推理的原逻辑） =====================
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

        return x

# ===================== 2. INT8全连接部分（保留原逻辑） =====================
class Int8FCPart(nn.Module):
    def __init__(self, fc1_npy, fc2_npy, fc3_npy):
        super().__init__()
        # 1. 加载并校验权重维度
        fc1_weight = np.load(fc1_npy).astype(np.float32)  # 直接转float32（int8转float不影响维度）
        fc2_weight = np.load(fc2_npy).astype(np.float32)  # 直接转float32（int8转float不影响维度）
        fc3_weight = np.load(fc3_npy).astype(np.float32)

        # 2. 定义全连接层并加载权重
        self.fc1 = nn.Linear(in_features=4608, out_features=512, bias=False)
        self.fc2 = nn.Linear(in_features=512, out_features=128, bias=False)
        self.fc3 = nn.Linear(in_features=128, out_features=NUM_CLASSES, bias=False)


        # 加载权重（转为torch tensor并移到设备）
        self.fc1.weight.data = torch.from_numpy(fc1_weight).to(DEVICE)
        self.fc2.weight.data = torch.from_numpy(fc2_weight).to(DEVICE)
        self.fc3.weight.data = torch.from_numpy(fc3_weight).to(DEVICE)
        
        # 3. 显式设置LIF参数，避免默认值冲突
        self.lif1 = neuron.LIFNode()
        self.lif2 = neuron.LIFNode()
        self.lif3 = neuron.LIFNode()

    def forward(self, x):
        x = self.fc1(x)
        x = self.lif1(x)
        x = self.fc2(x)
        x = self.lif2(x)
        x = self.fc3(x)
        x = self.lif3(x)
        return x

# ===================== 3. 整合推理模型（INT8 FC + 浮点卷积） =====================
class SNNInferModel(nn.Module):
    def __init__(self, model_path, fc1_npy, fc2_npy, fc3_npy, time_steps=TIME_STEPS):
        super().__init__()
        self.time_steps = time_steps
        self.float_part = FloatPart().to(DEVICE)
        self.int8_fc_part = Int8FCPart(fc1_npy, fc2_npy, fc3_npy).to(DEVICE)

        # 加载浮点卷积层权重
        state_dict = torch.load(model_path, map_location=DEVICE)
        # 兼容checkpoint格式（如果是带optimizer的checkpoint）
        if "model_state_dict" in state_dict:
            state_dict = state_dict["model_state_dict"]
        # 加载权重（strict=False忽略不匹配的层）
        self.float_part.load_state_dict(state_dict, strict=False)
        
        # 设为评估模式
        self.float_part.eval()
        self.int8_fc_part.eval()

    def forward(self, x):
        functional.reset_net(self.float_part)
        functional.reset_net(self.int8_fc_part)
        out_spk = 0.0
        x = self.float_part(x)
        x = x * 1.0/128.0
        # print(tmp.shape)
        # print(x[0][:20])
        for t in range(self.time_steps):
            # spk1 = self.float_part(x)
            out = self.int8_fc_part(x)
            out_spk += out
        return out_spk / self.time_steps

# ===================== 4. 事件检测核心函数 =====================
def detect_event_frame(img):
    """检测是否为有效事件帧（含扑克牌区域）"""
    b, g, r = cv2.split(img)
    white = (r > WHITE_THRESH) & (g > WHITE_THRESH) & (b > WHITE_THRESH)
    blue = (b > BLUE_THRESH) & (r < 100) & (g < 100)
    return (np.count_nonzero(white) + np.count_nonzero(blue)) >= (WHITE_THRESH + BLUE_THRESH)

def extract_index(name):
    """从文件名提取帧号（用于排序）"""
    digits = ''.join(filter(str.isdigit, name))
    return int(digits) if digits else -1

# ===================== 5. 单张图片预测（适配INT8模型） =====================
def predict_image(model, img_path):
    """单张图片预测（返回类别索引）"""
    try:
        img = Image.open(img_path).convert('RGB')
        img = test_transform(img).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            output = model(img)
            pred = torch.argmax(output, dim=1).item()
        return pred
    except Exception as e:
        print(f"⚠️  预测图片失败 {img_path}: {e}")
        return -1  # 异常返回-1

# ===================== 6. 事件级预测（投票逻辑） =====================
def predict_event(model, event_frames, base_dir):
    """对单个事件的帧列表进行投票预测"""
    votes = []
    for fname in event_frames:
        img_path = os.path.join(base_dir, fname)
        pred = predict_image(model, img_path)
        if pred != -1:  # 过滤异常帧
            votes.append(pred)

    if not votes:
        return -1, {}  # 无有效预测
    
    # 投票选出最终结果
    values, counts = np.unique(votes, return_counts=True)
    final_pred = values[np.argmax(counts)]
    vote_detail = dict(zip(values, counts))
    
    return final_pred, vote_detail

# ===================== 7. 主测试流程 =====================
def main():
    # 1. 加载INT8推理模型
    print("===== 加载INT8推理模型 =====")
    model = SNNInferModel(MODEL_PATH, FC1_WEIGHT_NPY, FC2_WEIGHT_NPY, FC3_WEIGHT_NPY)
    print("✅ 模型加载完成")

    # 2. 读取并排序事件帧
    print(f"\n===== 读取事件帧（{INPUT_DIR}） =====")
    files = [
        f for f in os.listdir(INPUT_DIR)
        if f.lower().endswith(('.jpg', '.png')) and not f.startswith('.DS_Store')
    ]
    files = sorted(files, key=extract_index)
    files = files[REMOVE_FIRST_N:]  # 跳过前N帧
    print(f"📁 有效帧数量: {len(files)}")

    # 3. 事件分割（按间隔阈值）
    print("\n===== 事件分割 =====")
    raw_events = []
    current_event = []
    gap = 0

    for fname in files:
        img_path = os.path.join(INPUT_DIR, fname)
        img = cv2.imread(img_path)
        if img is None:
            continue

        if detect_event_frame(img):
            current_event.append(fname)
            gap = 0
        else:
            gap += 1
            if gap >= MIN_GAP_FRAMES and current_event:
                raw_events.append(current_event)
                current_event = []

    # 处理最后一个事件
    if current_event:
        raw_events.append(current_event)

    # 4. 过滤有效事件（最小帧数量）
    valid_events = []
    for ev in raw_events:
        max_len = 0
        cur = 0
        for fname in ev:
            img = cv2.imread(os.path.join(INPUT_DIR, fname))
            if detect_event_frame(img):
                cur += 1
                max_len = max(max_len, cur)
            else:
                cur = 0
        if max_len >= MIN_EVENT_FRAMES:
            valid_events.append(ev)

    # 5. 输出事件统计
    print(f"\n===== 事件统计 =====")
    print(f"📊 原始事件数量: {len(raw_events)}")
    print(f"✅ 有效事件数量: {len(valid_events)}")
    for i, ev in enumerate(valid_events):
        print(f"Event {i+1}: {ev[0]} → {ev[-1]} ({len(ev)} frames)")

    # 6. 事件级预测（核心）
    print("\n===== INT8推理 - 事件级预测结果 =====")
    for idx, ev in enumerate(valid_events):
        if len(ev) <= 2:
            continue
        
        ev_core = ev[1:-1]  # 去掉首尾无效帧
        pred, vote_detail = predict_event(model, ev_core, INPUT_DIR)
        
        if pred == -1:
            print(f"Event {idx+1}: ❌ 无有效预测")
            continue
        
        # 转换为可读的投票结果
        readable_votes = {CLASS_NAMES[k]: int(v) for k, v in vote_detail.items()}
        print(f"Event {idx+1}: 🎯 预测结果 = {CLASS_NAMES[pred]}, 投票详情 = {readable_votes}")

if __name__ == "__main__":
    # 设置随机种子保证可复现
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    
    main()