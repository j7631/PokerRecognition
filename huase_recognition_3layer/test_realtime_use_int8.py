# ============================================================
# realtime_event_predict_int8.py
# 实时 DVS 事件 → RGB 累积 → 事件检测 → 浮点卷积 + INT8全连接 SNN 预测 → 投票
# 核心改造：模型拆分为浮点卷积 + INT8全连接两部分
# ============================================================

import os
import cv2 as cv
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from datetime import timedelta
from PIL import Image
from collections import Counter

import dv_processing as dv
from spikingjelly.activation_based import neuron, functional

from qat_layer import Quan_Linear

# ========================= 核心配置（新增INT8权重路径） =========================
REMOVE_FIRST_N = 10  # 跳过前10帧
MODEL_PATH = "conv+fc_5_class_best.pth"  # 浮点卷积层权重
FC2_WEIGHT_NPY = "/Users/bytedance/Desktop/bishe/2_huase_augment/save_int8_new/fc2_weight_int8.npy"  # INT8 FC2权重
FC3_WEIGHT_NPY = "/Users/bytedance/Desktop/bishe/2_huase_augment/save_int8_new/fc3_weight_int8.npy"  # INT8 FC3权重

CLASS_NAMES = ["empty", "fangpian", "heitao", "hongtao", "meihua"]
NUM_CLASSES = len(CLASS_NAMES)
TIME_STEPS = 3

TIME_INTERVAL_MS = 20
MIN_GAP_FRAMES = 10
MIN_EVENT_FRAMES = 7
WHITE_THRESH = 200
BLUE_THRESH = 200
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", DEVICE)

# ========================= 1. 浮点卷积特征提取部分 =========================
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
        self.fc1 = Quan_Linear(w_bits=8, in_features=128 * 6 * 6, out_features=512, bias=False)
        self.lif1 = neuron.LIFNode()

    def forward(self, x):
        x = self.conv_layer(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        spk1 = self.lif1(x)
        return spk1

# ===================== 2. INT8全连接部分（保留原逻辑） =====================
class Int8FCPart(nn.Module):
    def __init__(self, fc2_npy, fc3_npy):
        super().__init__()
        # 加载INT8权重并转为浮点
        fc2_weight = np.load(fc2_npy).astype(np.int8)
        fc3_weight = np.load(fc3_npy).astype(np.int8)

        self.fc2 = nn.Linear(in_features=512, out_features=128, bias=False)
        self.fc3 = nn.Linear(in_features=128, out_features=NUM_CLASSES, bias=False)


        # INT8 -> Float32 转换
        self.fc2.weight.data = torch.from_numpy(fc2_weight.astype(np.float32))
        self.fc3.weight.data = torch.from_numpy(fc3_weight.astype(np.float32))
        
        self.lif2 = neuron.LIFNode()
        self.lif3 = neuron.LIFNode()

    def forward(self, x):
        x = self.fc2(x)
        x = self.lif2(x)
        x = self.fc3(x)
        x = self.lif3(x)
        return x

# ===================== 3. 整合推理模型（INT8 FC + 浮点卷积） =====================
class SNNInferModel(nn.Module):
    def __init__(self, model_path, fc2_npy, fc3_npy, time_steps=TIME_STEPS):
        super().__init__()
        self.time_steps = time_steps
        self.float_part = FloatPart().to(DEVICE)
        self.int8_fc_part = Int8FCPart(fc2_npy, fc3_npy).to(DEVICE)

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
        functional.reset_net(self.float_part.lif1)
        out_spk = 0.0
        for t in range(self.time_steps):
            spk1 = self.float_part(x)
            out = self.int8_fc_part(spk1)
            out_spk += out
        return out_spk / self.time_steps

# ========================= 加载整合模型 =========================
model = SNNInferModel(MODEL_PATH, FC2_WEIGHT_NPY, FC3_WEIGHT_NPY).to(DEVICE)
model.eval()
print("✅ INT8推理模型加载完成！")

# ========================= 图像预处理 =========================
transform = transforms.Compose([
    transforms.Resize((48, 48)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ========================= 事件检测 =========================
def detect_event_frame(img):
    b, g, r = cv.split(img)
    white = (r > 200) & (g > 200) & (b > 200)
    blue = (b > 200) & (r < 100) & (g < 100)
    return (np.count_nonzero(white) + np.count_nonzero(blue)) >= (WHITE_THRESH + BLUE_THRESH)

def align_frame_like_offline(frame_bgr: np.ndarray):
    """将DVS输出帧转换为离线测试一致的格式"""
    # 动态范围压缩
    frame = cv.normalize(frame_bgr, None, 0, 255, cv.NORM_MINMAX)
    # uint8量化
    frame = frame.astype(np.uint8)
    # BGR → RGB → PIL
    frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    return Image.fromarray(frame_rgb)

# 类别映射（保留原有逻辑）
output_to_true = {
    0: 1,  # 模型fangpian -> 真heitao
    1: 2,  # 模型heitao -> 真hongtao
    2: 3,  # 模型hongtao -> 真meihua
    3: 4,  # 模型meihua -> 真empty
    4: 0   # 模型empty -> 真fangpian
}

# ========================= 帧级预测（适配INT8模型） =========================
@torch.no_grad()
def predict_frame(frame):
    img = align_frame_like_offline(frame)
    x = transform(img).unsqueeze(0).to(DEVICE)
    out = model(x)
    pred = torch.argmax(out, dim=1).item()
    pred_mapped = output_to_true[pred]  # 映射到真实类别
    return pred_mapped

# ========================= 打开相机 =========================
capture = dv.io.camera.open()
if not capture.isEventStreamAvailable():
    raise RuntimeError("Camera does not support event stream.")

visualizer = dv.visualization.EventVisualizer(capture.getEventResolution())
visualizer.setBackgroundColor(dv.visualization.colors.black())
visualizer.setPositiveColor(dv.visualization.colors.iniBlue())
visualizer.setNegativeColor(dv.visualization.colors.white())

filter_chain = dv.EventFilterChain()
filter_chain.addFilter(
    dv.noise.BackgroundActivityNoiseFilter(capture.getEventResolution())
)

cv.namedWindow("Preview", cv.WINDOW_NORMAL)

slicer = dv.EventStreamSlicer()

# ========================= 实时状态变量 =========================
frame_counter = 0  # 全局帧计数器，用于跳过前N帧
current_event_frames = []
current_event_preds = []
gap_counter = 0
event_id = 0

# ========================= 核心回调 =========================
def slicing_callback(events: dv.EventStore):
    global current_event_frames, gap_counter, frame_counter, current_event_preds, event_id

    # 事件滤波
    filter_chain.accept(events)
    filtered_events = filter_chain.generateEvents()
    frame = visualizer.generateImage(filtered_events)

    # 跳过前N帧
    frame_counter += 1
    if frame_counter <= REMOVE_FIRST_N:
        cv.imshow("Preview", frame)
        cv.waitKey(1)
        print(f"⏳ Skipping initial frame {frame_counter}/{REMOVE_FIRST_N}")
        return

    # 显示预览帧
    cv.imshow("Preview", frame)
    cv.waitKey(1)

    # 事件检测与预测
    if detect_event_frame(frame):
        pred = predict_frame(frame)
        current_event_frames.append(frame.copy())
        current_event_preds.append(pred)
        gap_counter = 0
    else:
        gap_counter += 1
        # 事件结束判定
        if gap_counter >= MIN_GAP_FRAMES and len(current_event_preds) > 0:
            event_id += 1
            # 提取核心预测结果（去掉首尾）
            core_preds = current_event_preds[1:-1] if len(current_event_preds) > 2 else current_event_preds
            
            if len(core_preds) > 0:
                final_pred = Counter(core_preds).most_common(1)[0][0]
                # 打印事件结果
                print(f"\n🎯 Event #{event_id}")
                print(f"   Final Prediction: {CLASS_NAMES[final_pred]}")
                print(f"   Vote Details: {Counter({CLASS_NAMES[k]:v for k,v in Counter(core_preds).items()})}")

                # UI显示结果
                disp_frame = frame.copy()
                cv.putText(
                    disp_frame,
                    f"Event #{event_id}: {CLASS_NAMES[final_pred]}",
                    (50, 50),
                    cv.FONT_HERSHEY_SIMPLEX,
                    1.2,
                    (0, 255, 0),
                    3
                )
                cv.imshow("Preview", disp_frame)
                cv.waitKey(1000)
            else:
                print(f"\n⚠️  Event #{event_id} skipped: no valid core predictions")

            # 重置事件状态
            current_event_frames = []
            current_event_preds = []
            gap_counter = 0

# ========================= 启动实时推理 =========================
slicer.doEveryTimeInterval(
    timedelta(milliseconds=TIME_INTERVAL_MS),
    slicing_callback
)

print("🚀 Real-time INT8 inference started!")

try:
    while capture.isRunning():
        events = capture.getNextEventBatch()
        if events is not None:
            slicer.accept(events)
except KeyboardInterrupt:
    print("\n🛑 Real-time inference stopped by user.")
finally:
    cv.destroyAllWindows()
    capture.stop()