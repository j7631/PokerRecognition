import os
os.chdir("/Users/bytedance/Desktop/bishe")

import paibox as pb
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from torchvision import transforms
from PIL import Image
from spikingjelly.activation_based import neuron, functional
from qat_layer import Quan_Linear
from paibox.components.neuron.neurons import StoreVoltageNeuron

from paiboard import PAIBoard_SIM
# from paiboard import PAIBoard_PCIe
# from paiboard import PAIBoard_Ethernet


# 超参数定义
NUM_CLASSES = 5
TIME_STEPS = 3
MODEL_PATH = '/Users/bytedance/Desktop/bishe/SNN/conv+fc_5_class_best.pth'
FC2_WEIGHT_NPY = '/Users/bytedance/Desktop/bishe/SNN/save_int8_new/fc2_weight_int8.npy'
FC3_WEIGHT_NPY = '/Users/bytedance/Desktop/bishe/SNN/save_int8_new/fc3_weight_int8.npy'
LABELS = ['fangpian', 'heitao', 'hongtao', 'meihua', 'empty']


# 新增：定义输入函数，返回多时间步输入数据
def snn_input_func(t, data):
    """
    timestep: 当前时间步
    data: 传入的多时间步数据（shape: [total_timesteps, input_dim]）
    返回：当前时间步的输入
    """
    timestep = data.shape[0]
    if t <= timestep and t >= 1:
        return data[t - 1]
    else:
        return np.zeros_like(data[0])

# 图片预处理
def image_preprocess(image_path):
    transform = transforms.Compose([
        transforms.Resize((48, 48)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                             std=[0.229, 0.224, 0.225])
    ])
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0)  # (1, 3, 48, 48)
    return image

# 加载权重
def get_weights():
    # 量化全连接权重（int8）
    fc2_weight = np.load(FC2_WEIGHT_NPY).astype(np.int8)
    fc3_weight = np.load(FC3_WEIGHT_NPY).astype(np.int8)
    
    # 浮点特征提取器权重
    float_state_dict = torch.load(MODEL_PATH, map_location='cpu')["model_state_dict"]
    
    return float_state_dict, fc2_weight, fc3_weight

# 浮点特征提取器（生成多时间步输入）
class FloatFeatureExtractor(nn.Module):
    def __init__(self, float_state_dict):
        super().__init__()
        self.conv_layer = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.fc1 = Quan_Linear(w_bits=8, in_features=128*6*6, out_features=512, bias=False)
        self.lif1 = neuron.LIFNode()
        
        self.load_state_dict(float_state_dict, strict=False)
        self.eval()

    def forward(self, x, timesteps=TIME_STEPS):
        """生成多时间步的输入脉冲（形状：[timesteps, 512]）"""
        functional.reset_net(self.lif1)
        x = self.conv_layer(x)
        x = torch.flatten(x, 1)  # (1, 128*6*6)
        x = self.fc1(x)  # (1, 512)
        
        # 生成TIME_STEPS步的脉冲
        spikes = []
        for _ in range(timesteps):
            spk = self.lif1(x)
            spikes.append(spk.detach().cpu().numpy().squeeze(0))  # 每个时间步形状：(512,)
        
        return np.stack(spikes, axis=0)  # 输出形状：(TIME_STEPS, 512)


class SNNInferNet(pb.Network):
    def __init__(self, float_state_dict, fc2_weight, fc3_weight):
        super().__init__()
        
        # 1. 浮点特征提取器
        self.float_extractor = FloatFeatureExtractor(float_state_dict)
        
        self.in_neuron = pb.InputProj(
            input=snn_input_func,  # 绑定输入函数
            shape_out=(512,)
        )
        
        # 2. PAIBox量化全连接网络（多时间步支持）
        self.n1 = pb.LIF((128,), threshold=1, reset_v=0)  # 中间层神经元
        self.n2 = pb.LIF((5,), threshold=0, reset_v=0)    # 输出层神经元
        
        # 全连接突触（权重已转置，适配PAIBox的All2All连接）
        self.s1 = pb.FullConn(self.in_neuron, self.n1, weights=fc2_weight, conn_type=pb.SynConnType.All2All)
        self.s2 = pb.FullConn(self.n1, self.n2, weights=fc3_weight, conn_type=pb.SynConnType.All2All)
        
        # 3. 最终输出层（nn.Linear + LIF）
        self.fc3 = nn.Linear(128, 5)
        fc3_weight_np = np.load(FC3_WEIGHT_NPY).astype(np.int8)
        self.fc3.weight.data = torch.from_numpy(fc3_weight_np.astype(np.float32))
        self.lif3 = neuron.LIFNode()
        
        # 4. Probe：监控s1输出（供fc3使用）和最终结果
        self.probe_s1 = pb.Probe(target=self.s1, attr="output")  # 形状：(TIME_STEPS, 128)
        self.probe_n2 = pb.Probe(target=self.n2, attr="voltage")  # 形状：(TIME_STEPS, 5)
        
        # 5. 初始化仿真器
        self.sim = pb.Simulator(self)

    def __call__(self, image_tensor, timestep=TIME_STEPS):
        """完全对齐标准格式：用sim.run(timestep)替代for循环"""
        # 步骤1：生成多时间步输入脉冲（形状：[timestep, 512]）
        input_spikes = self.float_extractor(image_tensor, timesteps=timestep)
        
        # 步骤2：PAIBox仿真（一次性运行所有时间步，无需手动循环）
        self.sim.reset()  # 重置仿真器
        # 传入多时间步输入数据，直接运行timestep步
        self.sim.run(timestep, reset=True, data=input_spikes)
        
        # 步骤3：提取s1的多时间步输出（形状：[timestep, 128]）
        s1_outputs = self.sim.data[self.probe_s1]  # (timestep, 128)
        
        # 步骤4：计算最终层输出（累加所有时间步）
        out_spk = 0.0
        for t in range(timestep):
            s1_tensor = torch.tensor(s1_outputs[t], dtype=torch.float32)
            fc3_out = self.fc3(s1_tensor)
            out_spk += self.lif3(fc3_out)
        
        # 时间步平均
        avg_spk = out_spk / timestep
        return avg_spk.detach().numpy()


def test():
    # 1. 加载权重
    float_state_dict, fc2_weight, fc3_weight = get_weights()
    print(f"权重加载完成：fc2形状={fc2_weight.shape}, fc3形状={fc3_weight.shape}")
    
    # 2. 初始化网络（权重转置适配全连接维度）
    net = SNNInferNet(float_state_dict, fc2_weight.T, fc3_weight.T)
    print("网络初始化完成")
    
    # 3. 测试图片预测
    test_images = {
        '/Users/bytedance/Desktop/bishe/data/fig2/hongtao/MipiData_20250223_131123422_F_100M/image_fullpic/20250223_131123_000053.jpg': 2,
        '/Users/bytedance/Desktop/bishe/data/fig2/hongtao/MipiData_20250223_131123422_F_100M/image_fullpic/20250223_131123_000007.jpg': 4,
        '/Users/bytedance/Desktop/bishe/data/fig1/heitao/MipiData_20250222_232938156_F_100M/image_fullpic/20250222_232938_000032.jpg': 1,
        '/Users/bytedance/Desktop/bishe/data/fig3/fangpian/MipiData_20250223_135552092_F_100M/image_fullpic/20250223_135552_000020.jpg': 0,
        '/Users/bytedance/Desktop/bishe/data/fig3/fangpian/MipiData_20250223_135601349_F_100M/image_fullpic/20250223_135601_000027.jpg': 0,
        '/Users/bytedance/Desktop/bishe/data/fig1/meihua/MipiData_20250222_230856290_F_100M/image_fullpic/20250222_230856_000042.jpg': 3,
    }
    
    # 4. 逐图预测
    for image_path, true_label in test_images.items():
        image_tensor = image_preprocess(image_path)
        pred_output = net(image_tensor)  # 直接调用网络，内部通过sim.run完成多时间步
        pred_label = np.argmax(pred_output)
        print(f'Image: {os.path.basename(image_path)}, '
              f'True: {LABELS[true_label]}, '
              f'Pred: {LABELS[pred_label]}, '
              f'Output: {pred_output.round(3)}')
    
    # # 5. PAIBox编译与导出
    # out_dir = Path(__file__).parent / "PAIBox_output"
    # out_dir.mkdir(exist_ok=True)
    
    # mapper = pb.Mapper()
    # mapper.build(net)  # 传入网络实例
    # graph_info = mapper.compile(
    #     weight_bit_optimization=True,
    #     grouping_optim_target="both"
    # )
    # print(f"编译完成，核心需求：{graph_info['n_core_required']}")
    
    # mapper.export(
    #     write_to_file=True,
    #     fp=out_dir,
    #     format="bin",
    #     split_by_chip=False,
    #     use_hw_sim=True
    # )
    # print(f"模型导出至：{out_dir}")
    # mapper.clear()

if __name__ == '__main__':
    test()

    timestep = 3
    layer_num = 0
    baseDir = "./added/output"
    snn = PAIBoard_SIM(baseDir, timestep, layer_num=layer_num)
    # snn = PAIBoard_PCIe(baseDir, timestep, layer_num=layer_num)
    # snn = PAIBoard_Ethernet(baseDir, timestep, layer_num=layer_num)
    snn.config(oFrmNum=1000)


    test_data=torch.load("./added/video/quan_test_data.pth")
    test_data_loader = data.DataLoader(
        dataset=test_data,
        batch_size=1,
        shuffle=True,
        num_workers=12,
        pin_memory=False
    )


    acc = 0
    num = 0
    for img,label in test_data_loader:

        input_data = img.resize(img.shape[1])# (130*173)
        input_spike = PoissonEncoder(x=input_data, timesteps=timestep)# (3,130*173)

        output_spike_dict = snn(input_spike, TimeMeasure=True)
        out1 = output_spike_dict["LIF_0"][-1]
        print(output_spike_dict["LIF_0"].shape)
        out2 = output_spike_dict["LIF_3"][0]
        print(output_spike_dict["LIF_3"])
        spike_out = np.concatenate((out2, out1), axis=0)
        print(label)
        print(spike_out)


        # pred = np.argmax(output_spike.sum(axis=0))
        # acc+=pred==label.item()
        # num+=1
    
    # print("acc:",acc/num)
    snn.perf(num)
