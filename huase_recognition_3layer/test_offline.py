import os
import cv2
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
import torch.nn as nn
from spikingjelly.activation_based import neuron, functional

# ======== 和训练代码保持一致 ========
IMG_SIZE = (48, 48)
NUM_CLASSES = 5
CLASS_NAMES = ['fangpian', 'heitao', 'hongtao', 'meihua', 'empty']

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ======== 测试用 transform（必须和训练一致） ========
test_transform = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                         std=[0.229, 0.224, 0.225])
])

# ======== 模型定义（原封不动复制） ========
from qat_layer import Quan_Linear

class ConvFCNet(nn.Module):
    def __init__(self, num_classes, time_steps=3):
        super().__init__()
        self.time_steps = time_steps

        self.conv_layer = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.fc1 = Quan_Linear(8, 128 * 6 * 6, 512, bias=False)
        self.lif1 = neuron.LIFNode()
        self.fc2 = Quan_Linear(8, 512, 128, bias=False)
        self.lif2 = neuron.LIFNode()
        self.fc3 = Quan_Linear(8, 128, num_classes, bias=False)
        self.lif3 = neuron.LIFNode()

    def forward(self, x):
        x = self.conv_layer(x)
        x = torch.flatten(x, 1)
        functional.reset_net(self)
        out = 0
        for _ in range(self.time_steps):
            spk1 = self.lif1(self.fc1(x))
            spk2 = self.lif2(self.fc2(spk1))
            spk3 = self.lif3(self.fc3(spk2))
            out += spk3
        return out / self.time_steps

# ======== 加载模型 ========
def load_model(weight_path):
    model = ConvFCNet(NUM_CLASSES).to(DEVICE)
    model.load_state_dict(torch.load(weight_path, map_location=DEVICE))
    model.eval()
    return model

# ======== 单张图片预测 ========
def predict_image(model, img_path):
    img = Image.open(img_path).convert('RGB')
    img = test_transform(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        out = model(img)
        pred = torch.argmax(out, dim=1).item()
    return pred

# ======== 事件级预测（核心） ========
def predict_event(model, event_frames, base_dir):
    votes = []
    for fname in event_frames:
        img_path = os.path.join(base_dir, fname)
        pred = predict_image(model, img_path)
        votes.append(pred)

    # 投票
    values, counts = np.unique(votes, return_counts=True)
    final_pred = values[np.argmax(counts)]

    return final_pred, dict(zip(values, counts))

# ======== 主测试逻辑 ========
def test_events(model, valid_events, base_dir):
    print("\n===== Event-level Prediction Results =====\n")
    for idx, ev in enumerate(valid_events):
        if len(ev) <= 2:
            continue

        ev_core = ev[1:-1]  # 去掉首尾
        pred, vote_detail = predict_event(model, ev_core, base_dir)

        readable_votes = {CLASS_NAMES[k]: int(v) for k, v in vote_detail.items()}
        print(f"Event {idx+1}: Predicted = {CLASS_NAMES[pred]}, Votes = {readable_votes}")

# ================== 使用示例 ==================
if __name__ == "__main__":
    MODEL_PATH = "conv+fc_5_class_best.pth"
    EVENT_DATA_DIR = "/Users/bytedance/Desktop/dvs_data/test_offline"

    import os
    import cv2
    import numpy as np

    # ================== 配置 ==================
    INPUT_DIR = "/Users/bytedance/Desktop/dvs_data/test_offline"  # 👈 你的新数据文件夹
    REMOVE_FIRST_N = 10

    WHITE_THRESH = 200
    BLUE_THRESH = 200
    MIN_GAP_FRAMES = 10
    MIN_EVENT_FRAMES = 7

    # ================== 事件判定 ==================
    def detect_event_frame(img):
        b, g, r = cv2.split(img)
        white = (r > 200) & (g > 200) & (b > 200)
        blue = (b > 200) & (r < 100) & (g < 100)
        return (np.count_nonzero(white) + np.count_nonzero(blue)) >= (WHITE_THRESH + BLUE_THRESH)

    # ================== 文件排序（按帧号） ==================
    def extract_index(name):
        digits = ''.join(filter(str.isdigit, name))
        return int(digits) if digits else -1

    # ================== 主流程 ==================
    files = [
        f for f in os.listdir(INPUT_DIR)
        if f.lower().endswith(('.jpg', '.png'))
    ]

    files = sorted(files, key=extract_index)

    # 去掉最前面 N 张
    files = files[REMOVE_FIRST_N:]

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

    # 处理结尾
    if current_event:
        raw_events.append(current_event)

    # ================== 过滤有效事件 ==================
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

    # ================== 输出结果 ==================
    print("\n================ 统计结果 ================\n")
    print(f"📁 数据路径: {INPUT_DIR}")
    print(f"📊 有效事件数量: {len(valid_events)}\n")

    for i, ev in enumerate(valid_events):
        print(f"Event {i+1}: {ev[0]}  →  {ev[-1]}  ({len(ev)} frames)")

    print("\n==========================================")



    model = load_model(MODEL_PATH)
    test_events(model, valid_events, EVENT_DATA_DIR)
