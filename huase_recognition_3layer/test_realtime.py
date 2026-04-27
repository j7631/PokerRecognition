# ============================================================
# realtime_event_predict.py
# 实时 DVS 事件 → RGB 累积 → 事件检测 → SNN 预测 → 投票
# ============================================================

import os
import cv2 as cv
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from datetime import timedelta
from qat_layer import Quan_Linear
from PIL import Image
from collections import Counter

import dv_processing as dv
from spikingjelly.activation_based import neuron, functional

REMOVE_FIRST_N = 10  # 新增：跳过前10帧

# ========================= 配置 =========================
MODEL_PATH = "conv+fc_5_class_best.pth"

CLASS_NAMES = ["empty", "fangpian", "heitao", "hongtao", "meihua"]

TIME_INTERVAL_MS = 20
MIN_GAP_FRAMES = 10
MIN_EVENT_FRAMES = 7
WHITE_THRESH = 200
BLUE_THRESH = 200
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", DEVICE)

# ========================= 模型定义（原封不动） =========================
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


# ========================= 加载模型 =========================
model = ConvFCNet(num_classes=len(CLASS_NAMES)).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()
print("Model loaded.")

# ========================= 图像预处理 =========================
# transform = transforms.Compose([
#     transforms.ToPILImage(),
#     transforms.Resize((48, 48)),
#     transforms.ToTensor(),
#     transforms.Normalize([0.485, 0.456, 0.406],
#                          [0.229, 0.224, 0.225])
# ])

transform = transforms.Compose([
    # 修复：先将OpenCV的BGR转为RGB，再转PIL
    # transforms.Lambda(lambda x: cv.cvtColor(x, cv.COLOR_BGR2RGB)),
    # transforms.ToPILImage(),
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
    """
    强制模拟：
    EventVisualizer → cv.imwrite → PIL.open
    """
    # 1️⃣ 动态范围压缩（关键）
    frame = cv.normalize(frame_bgr, None, 0, 255, cv.NORM_MINMAX)

    # 2️⃣ uint8 量化
    frame = frame.astype(np.uint8)

    # 3️⃣ BGR → RGB → PIL
    frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    return Image.fromarray(frame_rgb)
output_to_true = {
    0: 1,  # 模型fangpian -> 真heitao
    1: 2,  # 模型heitao -> 真hongtao
    2: 3,  # 模型hongtao -> 真meihua
    3: 4,  # 模型meihua -> 真empty
    4: 0   # 模型empty -> 真fangpian
}

@torch.no_grad()
# ========================= 帧级预测 =========================



# 修改 predict_frame
def predict_frame(frame):
    img = align_frame_like_offline(frame)
    x = transform(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        out = model(x)
        pred = torch.argmax(out, dim=1).item()
        pred_mapped = output_to_true[pred]   # 🔹映射到真实类别
    return pred_mapped



# ========================= 事件级投票 =========================
def predict_event(frames):
    votes = [predict_frame(f) for f in frames]
    values, counts = np.unique(votes, return_counts=True)
    final = values[np.argmax(counts)]
    return final, dict(zip(values, counts))

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
current_event_frames = []
gap_counter = 0
event_count = 0
frame_counter = 0  # 新增：全局帧计数器，用于跳过前N帧
current_event_frames=[]
current_event_preds=[]
gap_counter=0
event_id=0
# ========================= 核心回调 =========================
def slicing_callback(events: dv.EventStore):
    global current_event_frames, gap_counter, event_count, frame_counter,current_event_preds, event_id

    filter_chain.accept(events)
    filtered_events = filter_chain.generateEvents()
    frame = visualizer.generateImage(filtered_events)

    # 新增：跳过前10帧（核心修改）
    frame_counter += 1
    if frame_counter <= REMOVE_FIRST_N:
        cv.imshow("Preview", frame)
        cv.waitKey(1)
        print(f"⏳ Skipping initial frame {frame_counter}/{REMOVE_FIRST_N}")
        return  # 直接返回，不处理事件检测和预测

    # 前10帧跳过完成后，正常处理
    cv.imshow("Preview", frame)
    cv.waitKey(1)

    # if detect_event_frame(frame):
    #     current_event_frames.append(frame.copy())
    #     gap_counter = 0
    # else:
    #     gap_counter += 1

        # if gap_counter >= MIN_GAP_FRAMES and len(current_event_frames) > 0:
            # ===== 事件结束 =====
            # if len(current_event_frames) >= MIN_EVENT_FRAMES:
            #     core = current_event_frames[1:-1]
            #     if len(core) > 0:
            #         pred, votes = predict_event(core)
            #         event_count += 1

            #         print(
            #             f"\n🎯 Event #{event_count}\n"
            #             f"   Final: {CLASS_NAMES[pred]}\n"
            #             f"   Votes: "
            #             + ", ".join(
            #                 f"{CLASS_NAMES[k]}={v}"
            #                 for k, v in votes.items()
            #             )
            #         )
            #     else:
            #         print("\n⚠️  Event skipped: core frames empty after removing first/last")
            # else:
            #     print(f"\n⚠️  Event skipped: too few frames ({len(current_event_frames)} < {MIN_EVENT_FRAMES})")

            # 事件结束

            # current_event_frames=[]
            # current_event_preds=[]
            # gap_counter=0

            # current_event_frames = []
            # gap_counter = 0

    if detect_event_frame(frame):
        pred = predict_frame(frame)
        current_event_frames.append(frame.copy())
        current_event_preds.append(pred)
        gap_counter=0
    else:
        gap_counter+=1
        if gap_counter>=MIN_GAP_FRAMES and len(current_event_preds)>0:
            # 事件结束
            event_id+=1
            core_preds=current_event_preds[1:-1] if len(current_event_preds)>2 else current_event_preds
            final_pred=Counter(core_preds).most_common(1)[0][0]
            print(f"\n🎯 Event #{event_id} -> {CLASS_NAMES[final_pred]}")

            # UI显示最终事件
            disp_frame = frame.copy()
            cv.putText(disp_frame,f"Event #{event_id}: {CLASS_NAMES[final_pred]}",(50,50),
                    cv.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
            cv.imshow("Realtime DVS",disp_frame)
            cv.waitKey(1000)

            current_event_frames=[]
            current_event_preds=[]
            gap_counter=0
# ========================= 启动 =========================
slicer.doEveryTimeInterval(
    timedelta(milliseconds=TIME_INTERVAL_MS),
    slicing_callback
)

print("🚀 Real-time event prediction started.")

while capture.isRunning():
    events = capture.getNextEventBatch()
    if events is not None:
        slicer.accept(events)

cv.destroyAllWindows()
