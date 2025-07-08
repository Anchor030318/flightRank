# 航班排序DCN-v2模型

这是一个基于DCN-v2（Deep Cross Network v2）的航班排序解决方案，用于预测用户对不同航班选项的偏好并生成排名结果。

## 项目结构

```
flightRank/
├── main.py                    # 主入口文件
├── data_preprocessing.py      # 数据预处理模块
├── feature_engineering.py    # 特征工程模块
├── dcn_model.py              # DCN-v2模型定义
├── train_pipeline.py         # 训练管道
├── predict_pipeline.py       # 预测管道
├── submission_generator.py   # 提交文件生成器
├── requirements.txt          # 依赖包列表
├── README.md                 # 项目说明
├── train.parquet            # 训练数据
├── test.parquet             # 测试数据
├── train_analysis.txt       # 训练数据分析
└── test_analysis.txt        # 测试数据分析
```

## 功能特点

### 🔧 数据预处理 (`data_preprocessing.py`)
- **时间特征处理**: 提取小时、星期几、月份等时间特征
- **持续时间处理**: 将时间字符串转换为分钟数
- **分类特征编码**: 使用LabelEncoder处理航空公司、机场等分类特征
- **数值特征标准化**: 处理缺失值并进行标准化
- **布尔特征转换**: 将布尔值转换为数值

### ⚙️ 特征工程 (`feature_engineering.py`)
- **航班复杂度特征**: 转机次数、是否直飞、总航段数等
- **时间偏好特征**: 早班机、晚班机、黄金时段标识
- **价格相关特征**: 每小时价格、价格分级、税费比例等
- **航空公司特征**: 代码共享、航空公司数量统计
- **座位和行李特征**: 可用座位、行李额度统计
- **用户特征**: VIP状态、常旅客等级等
- **交互特征**: 价格与时间、用户类型的交互

### 🧠 DCN-v2模型 (`dcn_model.py`)
- **交叉网络**: 使用低秩分解实现高效的特征交叉
- **深度网络**: 多层全连接网络学习非线性关系
- **排序损失**: 结合二分类损失和组内排序损失
- **批归一化**: 加速训练并提高稳定性

### 🚀 训练管道 (`train_pipeline.py`)
- **完整训练流程**: 数据加载→预处理→特征工程→模型训练
- **早停机制**: 防止过拟合
- **学习率调度**: 自适应调整学习率
- **模型保存**: 自动保存最佳模型和预处理器

### 🔮 预测管道 (`predict_pipeline.py`)
- **批量预测**: 支持大规模数据的分批处理
- **排名生成**: 基于预测分数在组内生成排名
- **结果验证**: 确保排名的唯一性和连续性

### 📝 提交文件生成 (`submission_generator.py`)
- **格式验证**: 确保提交文件符合比赛要求
- **排名修复**: 自动修复重复或不连续的排名
- **交叉验证**: 与测试数据进行一致性检查
- **详细报告**: 生成预测统计和排名分布报告

## 安装和使用

### 1. 环境设置

```bash
# 安装依赖
pip install -r requirements.txt
```

### 2. 数据准备

确保以下文件在项目根目录：
- `train.parquet`: 训练数据
- `test.parquet`: 测试数据

### 3. 运行模型

#### 完整流程（推荐）
```bash
python main.py --mode all
```

#### 仅训练
```bash
python main.py --mode train
```

#### 仅预测（需要先训练）
```bash
python main.py --mode predict
```

### 4. 独立运行各模块

#### 数据预处理
```bash
python data_preprocessing.py
```

#### 特征工程
```bash
python feature_engineering.py
```

#### 模型训练
```bash
python train_pipeline.py
```

#### 预测
```bash
python predict_pipeline.py
```

#### 生成提交文件
```bash
python submission_generator.py
```

## 输出文件

### 训练阶段
- `best_model.pth`: 最佳模型权重
- `flight_ranking_pipeline_*.pkl`: 预处理器和特征工程器
- `training_history.png`: 训练历史图表

### 预测阶段
- `prediction_results.csv`: 详细预测结果
- `submission.csv`: 最终提交文件
- `submission_detailed.csv`: 详细提交结果

## 模型配置

可以在 `main.py` 中调整以下参数：

### 模型参数
```python
'model': {
    'cross_layers': 3,              # 交叉网络层数
    'cross_low_rank': 32,           # 低秩分解维度
    'deep_layers': [512, 256, 128], # 深度网络层数
    'dropout_rate': 0.3,            # Dropout比例
    'use_bn': True                  # 是否使用批归一化
}
```

### 训练参数
```python
'training': {
    'batch_size': 256,              # 批次大小
    'num_epochs': 50,               # 训练轮数
    'learning_rate': 0.001,         # 学习率
    'optimizer': 'adam',            # 优化器
    'weight_decay': 1e-5,           # 权重衰减
    'margin': 1.0,                  # 排序损失的边际
    'patience': 5,                  # 学习率调度耐心
    'early_stopping_patience': 10, # 早停耐心
    'num_workers': 4                # 数据加载进程数
}
```

## 数据格式

### 训练数据特征
- **用户信息**: `bySelf`, `companyID`, `isVip`, `frequentFlyer`等
- **航班信息**: `legs0_*`, `legs1_*` 包含航段详细信息
- **价格信息**: `totalPrice`, `taxes`, `miniRules*`等
- **目标变量**: `selected` (0或1表示是否被选择)

### 测试数据
- 与训练数据相同的特征，但没有 `selected` 字段

### 输出格式
```csv
Id,rank
18144679,1
18144680,2
18144681,3
...
```

## 技术特点

1. **深度特征交叉**: DCN-v2模型能够自动学习特征间的复杂交互
2. **低秩分解**: 降低模型复杂度，提高训练效率
3. **排序损失**: 专门针对排序任务设计的损失函数
4. **鲁棒预处理**: 全面的数据清洗和特征工程
5. **模块化设计**: 易于维护和扩展

## 性能优化建议

1. **增加训练轮数**: 如果验证损失还在下降，可以增加 `num_epochs`
2. **调整学习率**: 根据训练曲线调整 `learning_rate`
3. **特征选择**: 根据特征重要性选择最有效的特征
4. **模型集成**: 训练多个模型并进行集成
5. **超参数搜索**: 使用网格搜索或贝叶斯优化调参

## 注意事项

1. **内存使用**: 大数据集可能需要调整 `batch_size` 和 `num_workers`
2. **GPU支持**: 自动检测并使用GPU加速训练
3. **排名唯一性**: 系统会自动确保同组内排名不重复
4. **数据验证**: 提交前会进行多重验证确保格式正确

## 故障排除

### 常见问题

1. **内存不足**: 减少 `batch_size` 或 `num_workers`
2. **CUDA错误**: 确保PyTorch版本与CUDA版本兼容
3. **文件缺失**: 确保所有必要的数据文件都在正确位置
4. **权限错误**: 确保有写入当前目录的权限

### 调试模式

在各模块中设置 `debug=True` 可以获得更详细的日志信息。

## 贡献

欢迎提交Issue和Pull Request来改进这个项目！

## 许可证

此项目仅用于学习和研究目的。 