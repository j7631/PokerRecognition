import torch
import torch.nn as nn
from spikingjelly.activation_based import neuron, functional
from pathlib import Path
import numpy as np
from qat_layer import Quan_Linear  # 复用你的QAT量化层

# ===================== 复用QAT中的Round STE函数（和qat_layer.py保持一致） =====================
class Round(torch.autograd.Function):
    @staticmethod
    def forward(self, input):
        sign = torch.sign(input)
        output = sign * torch.floor(torch.abs(input) + 0.5)
        return output

    @staticmethod
    def backward(self, grad_output):
        grad_input = grad_output.clone()
        return grad_input

round_ste = Round.apply

# ===================== 模型定义（完全复用你的QAT模型） =====================
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
        # fc part: 复用你的Quan_Linear（保证和QAT一致）
        self.fc1 = Quan_Linear(w_bits=8,in_features = 128 * 6 * 6, out_features=512, bias=False)
        self.lif1 = neuron.LIFNode()
        self.fc2 = Quan_Linear(w_bits=8,in_features = 512, out_features=128, bias=False)
        self.lif2 = neuron.LIFNode()
        self.fc3 = Quan_Linear(w_bits=8,in_features = 128, out_features=5, bias=False)    
        self.lif3 = neuron.LIFNode()

    def forward(self, x):
        # 卷积提取特征（一次）
        x = self.conv_layer(x)
        x = torch.flatten(x, 1)

        # 初始化膜电位
        functional.reset_net(self)

        # 时间步循环
        out_spk = 0.0
        for t in range(self.time_steps):
            cur = self.fc1(x)
            spk1 = self.lif1(cur)
            cur2 = self.fc2(spk1)
            spk2 = self.lif2(cur2)
            cur3 = self.fc3(spk2)
            spk3 = self.lif3(cur3)
            out_spk += spk3

        # 输出为时间平均脉冲
        out_spk = out_spk / self.time_steps
        return out_spk

# ===================== 核心修改：匹配QAT的量化函数（替换原有简单量化） =====================
def quantize_to_int8_qat_compatible(tensor, w_bits=8):
    """
    完全匹配QAT中Quan_Linear的量化方式：
    1. Tanh归一化到[-1,1]
    2. 对称量化到[-2^(w_bits-1)+1, 2^(w_bits-1)-1]（8位时[-127,127]）
    3. 复用STE Round取整
    """
    # 步骤1：QAT中的Tanh归一化（和qat_layer.py完全一致）
    tensor_tanh = torch.tanh(tensor)
    max_abs_tanh = torch.max(torch.abs(tensor_tanh))
    tensor_norm = tensor_tanh / max_abs_tanh  # 归一化到[-1,1]

    # 步骤2：计算QAT中的scale（和qat_layer.py一致）
    scale = 1 / float(2 ** (w_bits - 1) - 1)  # 8位时scale=1/127≈0.007874

    # 步骤3：量化（复用STE Round取整，和QAT一致）
    tensor_quant_norm = round_ste(tensor_norm / scale)  # 取整后范围[-127,127]
    # 截断到有效范围（避免溢出）
    tensor_quant_norm_clamp = tensor_quant_norm.clamp(
        -(2 ** (w_bits - 1) - 1),
        2 ** (w_bits - 1) - 1
    )
    # 转为INT8
    q_weight = tensor_quant_norm_clamp.to(torch.int8)

    # 返回量化权重 + scale + Tanh归一化的max_abs（推理反量化需要）
    return q_weight, scale, max_abs_tanh

# ===================== 实例化模型 =====================
model = ConvFCNet(num_classes=5, time_steps=3)
model.eval()

# ===================== 打印量化前权重示例（对比QAT训练后的权重） =====================
print("========== Before Quantization (QAT-trained) ==========")
for name, param in model.named_parameters():
    if 'weight' in name:
        print(f"{name:25} | min={param.min():.6f}, max={param.max():.6f} | dtype={param.dtype}")
        print("Sample values:", param.flatten()[:10].tolist())  # 打印前10个值
        print("-"*50)

# ===================== 对 fc2 和 fc3 执行匹配QAT的INT8量化 =====================
quantized_weights = {}
for layer_name in ['fc2', 'fc3']:
    weight = getattr(model, layer_name).weight.data
    # 使用匹配QAT的量化函数
    q_weight, scale, max_abs_tanh = quantize_to_int8_qat_compatible(weight, w_bits=8)
    quantized_weights[layer_name] = (q_weight, scale, max_abs_tanh)

    print(f"\n========== After Quantization (QAT-compatible): {layer_name} ==========")
    print(f"Quantized int8 weight: min={q_weight.min()}, max={q_weight.max()}")
    print(f"QAT scale: {scale:.6f}, Tanh max_abs: {max_abs_tanh:.6f}")
    print("Sample int8 values:", q_weight.flatten()[:10].tolist())

# ===================== 保存量化后的INT8权重（适配你的保存逻辑） =====================
def save_fc_int8_weights(model_path, npy_dir, w_bits=8):
    # 1. 加载原始QAT训练后的模型
    model = ConvFCNet(num_classes=5, time_steps=3)
    # 兼容checkpoint格式（带model_state_dict）
    checkpoint = torch.load(model_path, map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    else:
        model.load_state_dict(checkpoint, strict=False)
    model.eval()

    # 2. 确保保存目录存在
    Path(npy_dir).mkdir(parents=True, exist_ok=True)

    # 3. 对fc2/fc3执行匹配QAT的量化并保存
    fc_layers = ['fc1']
    for name in fc_layers:
        layer = getattr(model, name)
        weight = layer.weight.data
        bias = layer.bias.data if layer.bias is not None else None

        # 使用匹配QAT的量化函数
        q_weight, scale, max_abs_tanh = quantize_to_int8_qat_compatible(weight, w_bits=w_bits)

        # 保存INT8权重
        weight_path = Path(npy_dir) / f"{name}_weight_int8.npy"
        np.save(weight_path, q_weight.cpu().numpy())
        print(f"\n{name} weight saved to {weight_path}")
        print(f"  - shape: {q_weight.shape} | scale (QAT): {scale:.6f} | Tanh max_abs: {max_abs_tanh:.6f}")

        # 保存scale和max_abs（推理反量化需要）
        scale_path = Path(npy_dir) / f"{name}_scale.npy"
        max_abs_path = Path(npy_dir) / f"{name}_max_abs_tanh.npy"
        np.save(scale_path, np.array(scale))
        np.save(max_abs_path, np.array(max_abs_tanh.cpu()))
        print(f"  - scale saved to {scale_path}")
        print(f"  - tanh max_abs saved to {max_abs_path}")

        if bias is not None:
            bias_path = Path(npy_dir) / f"{name}_bias.npy"
            np.save(bias_path, bias.cpu().numpy())
            print(f"  - bias saved to {bias_path} | shape: {bias.shape}")
        else:
            print(f"  - {name} has no bias (QAT setting), skipped.")

# ===================== 执行保存 =====================
if __name__ == "__main__":
    model_path = "/Users/bytedance/Desktop/bishe/2_huase_augment/conv+fc_5_class_best.pth"  # 你的QAT模型路径
    npy_dir = "save_int8_qat_compatible"  # 保存目录（区分原有量化结果）
    save_fc_int8_weights(model_path, npy_dir, w_bits=8)