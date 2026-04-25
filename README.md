# 项目README

## 项目流程 (更新时间: 2026.01.15)

以"花色"任务为例，路径在 `2_huase_augment` 里

### 1. 训练与测试流程

1. **运行main函数**：得到QAT的训练结果并可以实时观察（需要dvs的conda环境）
2. **运行test_val**：查看保存模型在测试集上的准确率【nn.linear是随机输出，但是改成Qlinear就可以】
3. **运行quantize**：得到量化int8的权重
4. **运行test_val_use_int8**：测试在测试集上使用int8权重的效果【这里用nn linear和q linear效果一样，acc都很高】（直接点三角运行不要进环境）

### 2. 离线测试（8张扑克牌）

- **test_offline**：查看保存权重在原始网络的输出【只能用q linear】
- **test_offline_use_int8**：查看量化后的权重的输出【用nn linear和q linear效果一样】

### 3. 在线测试（实时读取摄像头）

需要进dvs环境：
- **test_realtime**
- **test_realtime_use_int8**

### 4. 编译与仿真

- 开始编译，把int8部分的网络用相关代码来运行
- 仿真【暂未跑过，但是不影响后续】

### 5. 板级测试

进入board文件夹：
- **board_huase_val**：观察在所有测试集上的结果（直接点三角运行不要进环境，下面同理）
- **board_huase_offline**：观察8张扑克牌事件的预测结果
- **board_huase_realtime**：实时测试

## 项目结构

- `2_huase_augment`：花色任务的主要代码
- `SNN`：包含模型权重和量化后的权重
- `bianyi.ipynb`：包含项目的主要代码和测试结果
- `qat_layer.py`：量化层的实现
- `requirements.txt`：环境依赖

## 环境要求

- Python 3.x
- PyTorch
- spikingjelly
- PAIBox
- OpenCV（用于实时测试）
