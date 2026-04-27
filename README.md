# 项目README

## 项目流程 

以"花色"任务为例，路径在 `2_huase_augment` 里

简略版运行流程：（按顺序依次运行）

/suit_project/huase_recognition/main.py

/suit_project/huase_recognition/quantize.py

/suit_project/PAIBOX_new/bianyi.py

/suit_project/PAIBoard/board_huase_dianshu_realtime.py


### 1. 训练与测试流程

1. **运行main函数**：得到QAT的训练结果并可以实时观察
2. **运行test_val**：查看保存模型在测试集上的准确率
3. **运行quantize**：得到量化int8的权重
4. **运行test_val_use_int8**：测试在测试集上使用int8权重的效果

### 2. 离线测试（8张扑克牌）

- **test_offline**：查看保存权重在原始网络的输出
- **test_offline_use_int8**：查看量化后的权重的输出

### 3. 在线测试（实时读取摄像头）

- **test_realtime**
- **test_realtime_use_int8**

### 4. 编译与仿真

- 开始编译，把int8部分的网络用相关代码来运行 bianyi.py
- 仿真

### 5. 板级测试

进入board文件夹：
- **board_huase_val**：观察在所有测试集上的结果
- **board_huase_offline**：观察8张扑克牌事件的预测结果
- **board_huase_realtime**：实时测试

## 项目结构

- `huase_recognition`：花色任务的主要代码
- `SNN`：包含模型权重和量化后的权重
- `PAIBOX_new`：PAIBox工具库，用于模型编译和优化
- `paiboard_new`：板级测试相关代码
- `bianyi.py`：包含项目的主要代码和测试结果
- `qat_layer.py`：量化层的实现
- `requirements.txt`：环境依赖

## 环境要求

- Python 3.x
- PyTorch
- spikingjelly
- PAIBox
- OpenCV（用于实时测试）
