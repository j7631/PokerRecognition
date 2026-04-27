# ============================================================
# realtime_save_and_predict.py
# 修复纯黑帧问题：增加事件累积+关闭过度降噪+调试模式
# ============================================================
import os
import cv2 as cv
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from datetime import timedelta, datetime
import dv_processing as dv
from spikingjelly.activation_based import neuron, functional
import time
from collections import Counter
from PIL import Image

# ========================= 核心配置 =========================
# 路径配置
SAVE_DIR = "/Users/bytedance/Desktop/dvs_data/test_realtime"  # 实时保存帧的目录
MODEL_PATH = "conv+fc_5_class_best.pth"   # 模型路径
os.makedirs(SAVE_DIR, exist_ok=True)

# 事件检测配置（和离线版完全一致）
REMOVE_FIRST_N = 10       # 跳过前10帧
WHITE_THRESH = 200
BLUE_THRESH = 200
MIN_GAP_FRAMES = 10       # 事件间隔帧数
MIN_EVENT_FRAMES = 7      # 有效事件最小帧数

# 模型配置
CLASS_NAMES = ['fangpian', 'heitao', 'hongtao', 'meihua', 'empty']  # 和训练一致
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 实时配置（关键修改：增加事件累积时间）
TIME_INTERVAL_MS = 50     # 从20ms→50ms，增加单帧事件累积量
SCAN_INTERVAL_S = 1       # 定时扫描新增帧的间隔（秒）
LAST_PROCESSED_FRAME = 0  # 记录最后处理的帧ID（避免重复处理）
DEBUG_MODE = True         # 调试模式：打印事件数量/帧信息

# ========================= 模型定义（和训练一致） =========================
try:
    from qat_layer import Quan_Linear  # 保持量化层一致
except ImportError:
    # 兼容：如果没有量化层，用普通Linear（保证代码能运行）
    print("⚠️  未找到qat_layer.py，使用普通Linear层")
    class Quan_Linear(nn.Linear):
        def __init__(self, w_bits, in_features, out_features, bias=False):
            super().__init__(in_features, out_features, bias)

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

        out = 0.0
        for _ in range(self.time_steps):
            spk1 = self.lif1(self.fc1(x))
            spk2 = self.lif2(self.fc2(spk1))
            spk3 = self.lif3(self.fc3(spk2))
            out += spk3

        return out / self.time_steps

# ========================= 加载模型 =========================
try:
    model = ConvFCNet(num_classes=len(CLASS_NAMES)).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    print(f"✅ 模型加载完成，使用设备：{DEVICE}")
except Exception as e:
    print(f"⚠️  模型加载失败：{e}，将使用随机初始化模型（仅保证运行）")
    model = ConvFCNet(num_classes=len(CLASS_NAMES)).to(DEVICE)
    model.eval()

# ========================= 预处理（和离线版一致） =========================
test_transform = transforms.Compose([
    transforms.Resize((48, 48)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                         std=[0.229, 0.224, 0.225])
])

# ========================= 复用离线版核心函数 =========================
def extract_index(name):
    """从文件名提取帧号（frame_0012.jpg → 12）"""
    digits = ''.join(filter(str.isdigit, name))
    return int(digits) if digits else -1

def detect_event_frame(img_path):
    """检测是否为事件帧（复用离线逻辑）"""
    img = cv.imread(img_path)
    if img is None:
        return False
    # 调试：打印帧的像素均值（纯黑帧均值接近0）
    if DEBUG_MODE and extract_index(img_path) % 10 == 0:
        print(f"📊 帧{extract_index(img_path)}像素均值：{np.mean(img)}")
    b, g, r = cv.split(img)
    white = (r > WHITE_THRESH) & (g > WHITE_THRESH) & (b > WHITE_THRESH)
    blue = (b > BLUE_THRESH) & (r < 100) & (g < 100)
    return (np.count_nonzero(white) + np.count_nonzero(blue)) >= (WHITE_THRESH + BLUE_THRESH)

def predict_image(model, img_path):
    """单张图片预测（复用离线逻辑）"""
    img = Image.open(img_path).convert('RGB')
    img = test_transform(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        out = model(img)
        pred = torch.argmax(out, dim=1).item()
    return pred

def predict_event(model, event_frames, base_dir):
    """事件级投票（复用离线逻辑）"""
    votes = []
    for fname in event_frames:
        img_path = os.path.join(base_dir, fname)
        pred = predict_image(model, img_path)
        votes.append(pred)

    values, counts = np.unique(votes, return_counts=True)
    final_pred = values[np.argmax(counts)]
    return final_pred, dict(zip(values, counts))

def scan_and_process_new_frames():
    """定时扫描新增帧，按离线逻辑处理事件"""
    global LAST_PROCESSED_FRAME

    # 1. 获取所有帧文件并排序
    files = [
        f for f in os.listdir(SAVE_DIR)
        if f.lower().endswith(('.jpg', '.png')) and extract_index(f) > LAST_PROCESSED_FRAME
    ]
    if not files:
        return  # 无新增帧，直接返回

    # 2. 排序并跳过前10帧（全局跳过）
    files = sorted(files, key=extract_index)
    if extract_index(files[0]) <= REMOVE_FIRST_N:
        files = [f for f in files if extract_index(f) > REMOVE_FIRST_N]
        if not files:
            LAST_PROCESSED_FRAME = extract_index(files[-1]) if files else LAST_PROCESSED_FRAME
            return

    # 3. 复用离线版事件检测逻辑
    raw_events = []
    current_event = []
    gap = 0

    for fname in files:
        img_path = os.path.join(SAVE_DIR, fname)
        if detect_event_frame(img_path):
            current_event.append(fname)
            gap = 0
        else:
            gap += 1
            if gap >= MIN_GAP_FRAMES and current_event:
                raw_events.append(current_event)
                current_event = []

    # 处理最后一个未完成事件
    if current_event:
        raw_events.append(current_event)

    # 4. 过滤有效事件并预测
    valid_events = []
    for ev in raw_events:
        max_len = 0
        cur = 0
        for fname in ev:
            img_path = os.path.join(SAVE_DIR, fname)
            if detect_event_frame(img_path):
                cur += 1
                max_len = max(max_len, cur)
            else:
                cur = 0
        if max_len >= MIN_EVENT_FRAMES:
            valid_events.append(ev)

    # 5. 输出预测结果
    for idx, ev in enumerate(valid_events):
        if len(ev) <= 2:
            continue
        ev_core = ev[1:-1]  # 去掉首尾
        pred, vote_detail = predict_event(model, ev_core, SAVE_DIR)
        readable_votes = {CLASS_NAMES[k]: int(v) for k, v in vote_detail.items()}
        
        # 打印结果（标注帧范围）
        start_idx = extract_index(ev[0])
        end_idx = extract_index(ev[-1])
        print(f"\n🎯 检测到新事件 {idx+1} (帧{start_idx}-{end_idx})")
        print(f"   最终预测：{CLASS_NAMES[pred]}")
        print(f"   投票结果：{readable_votes}")

    # 6. 更新最后处理的帧ID（避免重复处理）
    if files:
        LAST_PROCESSED_FRAME = extract_index(files[-1])

# ========================= 实时保存DVS帧（核心修复） =========================
def save_dvs_frames():
    """实时从DVS摄像头获取事件并保存帧（修复纯黑问题）"""
    # 打开摄像头
    capture = dv.io.camera.open()
    if not capture.isRunning():
        raise RuntimeError("❌ 无法打开摄像头，请检查设备连接！")
    if not capture.isEventStreamAvailable():
        raise RuntimeError("❌ 摄像头不提供事件流！")
    
    # 获取摄像头分辨率（关键：确保可视化器分辨率正确）
    cam_res = capture.getEventResolution()
    # print(f"📷 摄像头分辨率：{cam_res.width}x{cam_res.height}")

    # 初始化可视化器（关键修改：调整累积模式）
    visualizer = dv.visualization.EventVisualizer(cam_res)
    # 增加事件累积权重，让少量事件也能显示
    # visualizer.setEventRenderingWeight(5.0)
    # 调整颜色（确保事件可见）
    visualizer.setBackgroundColor(dv.visualization.colors.black())
    visualizer.setPositiveColor(dv.visualization.colors.red())  # 正极性事件设为红色（更明显）
    visualizer.setNegativeColor(dv.visualization.colors.white()) # 负极性事件设为白色

    # 事件过滤（关键修改：关闭过度降噪，仅保留基础过滤）
    filter_chain = dv.EventFilterChain()
    # 注释掉降噪过滤器（避免过滤所有事件）
    # filter_chain.addFilter(dv.noise.BackgroundActivityNoiseFilter(cam_res))

    # 初始化切片器
    slicer = dv.EventStreamSlicer()
    frame_id = 0

    # 切片回调：保存帧（增加调试信息）
    def slicing_callback(events: dv.EventStore):
        nonlocal frame_id
        # 调试：打印事件数量
        if DEBUG_MODE:
            print(f"📥 切片{frame_id}：收到{len(events)}个事件")
        
        # 过滤事件（仅基础过滤）
        filter_chain.accept(events)
        filtered_events = filter_chain.generateEvents()
        
        # 调试：打印过滤后的事件数量
        if DEBUG_MODE:
            print(f"🔍 切片{frame_id}：过滤后{len(filtered_events)}个事件")
        
        # 生成可视化帧（关键：强制刷新，避免累积不足）
        frame = visualizer.generateImage(filtered_events)
        # 调试：打印帧的像素均值（纯黑帧均值接近0）
        if DEBUG_MODE and frame_id % 5 == 0:
            frame_np = frame.numpy()
            print(f"📊 帧{frame_id}像素均值：{np.mean(frame_np)}")
        
        # 保存帧（命名：frame_0001.jpg）
        save_path = os.path.join(SAVE_DIR, f"frame_{frame_id:04d}.jpg")
        # 转换为OpenCV格式并保存
        cv.imwrite(save_path, frame.numpy())
        
        # 实时预览（放大窗口，便于观察）
        preview_frame = cv.resize(frame.numpy(), (640, 480))
        cv.putText(preview_frame, f"Frame: {frame_id} | Events: {len(filtered_events)}", 
                   (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv.imshow("DVS Preview (Red=+ Event, White=- Event)", preview_frame)
        cv.waitKey(1)
        
        frame_id += 1

    # 注册回调（关键：增加事件累积时间到50ms）
    slicer.doEveryTimeInterval(timedelta(milliseconds=TIME_INTERVAL_MS), slicing_callback)

    print(f"\n🚀 开始实时保存DVS帧到：{SAVE_DIR}")
    print(f"🔧 配置：每{TIME_INTERVAL_MS}ms保存一帧 | 每{SCAN_INTERVAL_S}秒扫描事件")
    print(f"💡 提示：移动物体让DVS产生事件，预览窗口会显示红/白色事件\n")
    
    # 主循环：保存帧 + 定时扫描
    try:
        while capture.isRunning():
            # 保存帧
            events = capture.getNextEventBatch()
            if events is not None:
                slicer.accept(events)
            
            # 定时扫描新增帧（每1秒）
            time.sleep(SCAN_INTERVAL_S)
            scan_and_process_new_frames()
            
            # 按ESC退出
            if cv.waitKey(1) & 0xFF == 27:
                break
    except KeyboardInterrupt:
        print("\n🛑 用户终止程序")
    finally:
        # capture.stop()
        cv.destroyAllWindows()
        print(f"\n✅ 程序结束，共保存 {frame_id} 帧到 {SAVE_DIR}")
        # 调试：打印最后10帧的像素均值
        if DEBUG_MODE and frame_id > 0:
            print("\n📊 最后10帧像素均值：")
            for i in range(max(0, frame_id-10), frame_id):
                img_path = os.path.join(SAVE_DIR, f"frame_{i:04d}.jpg")
                if os.path.exists(img_path):
                    img = cv.imread(img_path)
                    print(f"   帧{i}：{np.mean(img):.2f}")

# ========================= 启动程序 =========================
if __name__ == "__main__":
    # 清空历史帧（可选，避免干扰）
    for f in os.listdir(SAVE_DIR):
        if f.lower().endswith(('.jpg', '.png')):
            os.remove(os.path.join(SAVE_DIR, f))
    print("🗑️  清空历史帧文件")
    
    # 启动实时保存+扫描
    save_dvs_frames()