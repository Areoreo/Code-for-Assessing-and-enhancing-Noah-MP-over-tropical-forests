# LSTM Emulator for Noah-MP Calibration

基于LSTM神经网络的Noah-MP陆面模型快速参数校准系统。

## 工作流概览

```
┌─────────────────────────────────────────────────────────────────────┐
│                     run_model_training.sh                           │
│  1. 生成参数样本 (LHS)                                               │
│  2. 提取forcing数据 (full/calibration/validation时段)               │
│  3. 运行Noah-MP (1000组参数 × 全时段)                                │
│  4. 预处理数据 (calibration时段)                                     │
│  5. 训练LSTM emulator                                               │
└─────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────┐
│                       run_calibration.sh                            │
│  1. 使用emulator进行参数校准 (calibration时段)                       │
│  2. 用校准参数运行Noah-MP验证 (全时段)                               │
│  3. 分析并绘图 (区分calibration/validation时段)                      │
└─────────────────────────────────────────────────────────────────────┘
```

## 快速开始

```bash
cd /home/petrichor/ymwang/snap/Emulator-based_calibration/calibration-BCI

# 完整模型训练流程
bash run_model_training.sh --samples 1000 --parallel 4

# 完整校准流程
bash run_calibration.sh --num_calibration 10 --max_iter 100
```

## 时间范围配置

在 `config_forward_comprehensive.py` 中定义：

| 时段 | 范围 | 用途 |
|------|------|------|
| FULL | 2015-07-30 ~ 2017-07-30 | Noah-MP运行、验证 |
| CALIBRATION | 2015-07-30 ~ 2016-07-29 | emulator训练、参数校准 |
| VALIDATION | 2016-07-30 ~ 2017-07-30 | 独立验证 |

## 详细步骤

### 模型训练 (`run_model_training.sh`)

```bash
# 完整运行
bash run_model_training.sh

# 跳过已完成步骤
bash run_model_training.sh --skip-param-gen --skip-forcing
bash run_model_training.sh --skip-noahmp
bash run_model_training.sh --skip-preprocess --skip-train

# 可选参数
--samples N      # 参数样本数 (默认1000)
--parallel N     # 并行数 (默认4)
```

**输出文件：**
- `data/raw/param/noahmp_param_sets.txt` - 参数样本
- `data/raw/forcing/forcing_panama_*.nc` - forcing数据
- `data/raw/sim_results/sample_*/` - Noah-MP输出
- `data/processed_data_forward_comprehensive.pkl` - 训练数据
- `results_forward_comprehensive/*/` - 训练好的模型

### 参数校准 (`run_calibration.sh`)

```bash
# 完整运行
bash run_calibration.sh

# 指定模型目录
bash run_calibration.sh --model_dir results_forward_comprehensive/AttentionLSTM_xxx/

# 调整校准参数
bash run_calibration.sh --num_calibration 20 --max_iter 200

# 跳过步骤
bash run_calibration.sh --skip-calibration  # 使用已有校准结果
bash run_calibration.sh --skip-noahmp       # 跳过验证运行
```

**输出文件：**
- `calibration_results/*/calibration_1/` - 校准结果
  - `calibrated_parameters.csv` - 校准后参数
  - `calibration_result.json` - 详细metrics
- `validation_results/*/` - 验证结果
  - `validation_metrics_by_period.csv` - 分时段metrics
  - `timeseries_*.png/pdf` - 时序对比图
  - `scatter_*.png/pdf` - 散点图
  - `metrics_*.png/pdf` - metrics对比图

## SATDK参数处理

SATDK参数在不同阶段使用不同形式：

| 阶段 | 值形式 | 说明 |
|------|--------|------|
| 参数样本生成 | 原始值 | `[8.9e-06, 5e-04]` |
| TBL文件生成 | 原始值 | 直接写入Noah-MP配置 |
| emulator训练 | log10值 | 自动转换 `→ [-5.05, -3.3]` |
| 参数校准 | log10值 | 使用`SATDK(log)` bounds |
| 校准结果转换 | log→原始 | `10^x` 转换回物理值 |

**value_bounds.csv 格式：**
```csv
variable,Lower bound,Upper bound
SATDK,8.90E-06,5.00E-04
SATDK(log),-5.05,-3.3
```

## 单独运行各步骤

### 生成参数样本
```bash
cd noahmp/TBL_generator
python3 generate_samples.py
```

### 提取forcing数据
```bash
# 全时段
python3 extract_forcing_data.py --daily \
    --start_date 2015-07-30 --end_date 2017-07-30 \
    --output data/raw/forcing/forcing_panama_daily_full.nc

# 校准时段
python3 extract_forcing_data.py --daily \
    --start_date 2015-07-30 --end_date 2016-07-29 \
    --output data/raw/forcing/forcing_panama_daily_calibration.nc
```

### 数据预处理
```bash
# 全时段
python3 01_data_preprocessing_forward_comprehensive.py

# 校准时段 (用于训练emulator)
python3 01_data_preprocessing_forward_comprehensive.py --calibration
```

### 训练emulator
```bash
python3 02_train_forward_comprehensive.py
```

### 参数校准
```bash
python3 06_calibration_applying_emulator_multiple_runs.py \
    --model_dir results_forward_comprehensive/AttentionLSTM_xxx/ \
    --forcing data/raw/forcing/forcing_panama_daily_calibration.nc \
    --obs data/obs/Panama_BCI_obs_2015-07-30_2016-07-29.csv \
    --bounds value_bounds.csv \
    --num_calibration 10 \
    --max_iter 100
```

## 变量说明

### Forcing变量 (8个)
| 变量 | 描述 |
|------|------|
| T2D | 2m气温 (K) |
| Q2D | 2m比湿 (kg/kg) |
| PSFC | 地表气压 (Pa) |
| U2D, V2D | 2m风速分量 (m/s) |
| LWDOWN | 下行长波辐射 (W/m²) |
| SWDOWN | 下行短波辐射 (W/m²) |
| RAINRATE | 降水率 (mm/s) |

### 目标变量 (29个)
- **能量平衡** (5): FSA, FIRA, HFX, LH, GRDFLX
- **水通量** (5): ECAN, ETRAN, EDIR, UGDRNOFF_RATE, SFCRNOFF_RATE
- **水储量** (5): SOIL_M (L1-L4), CANLIQ
- **温度** (5): SOIL_T (L1-L2), TG, TV, TRAD
- **能量分量** (9): SAV, SAG, IRC, IRG, SHC, SHG, EVC, EVG, GHV

### 校准参数 (9个)
| 参数 | 描述 | 范围 |
|------|------|------|
| VCMX25 | 最大羧化速率 | [30, 120] |
| HVT | 冠层顶高 | [9, 55] |
| HVB | 冠层底高 | [0.1, 15] |
| CWPVT | 冠层风参数 | [0.15, 5.35] |
| Z0MVT | 动量粗糙度 | [0.3, 2] |
| WLTSMC | 凋萎含水量 | [0.02, 0.26] |
| REFSMC | 参考含水量 | [0.15, 0.45] |
| MAXSMC | 饱和含水量 | [0.45, 0.75] |
| SATDK | 饱和导水率 | [8.9e-6, 5e-4] |

## 模型配置

编辑 `config_forward_comprehensive.py`：

```python
TEMPORAL_RESOLUTION = 'daily'  # 'daily' 或 '30min'

MODEL_CONFIG = {
    'model_type': 'AttentionLSTM',
    'hidden_dim': 1536,
    'num_layers': 3,
    'dropout': 0.3,
}

TRAINING_CONFIG = {
    'learning_rate': 0.0005,
    'batch_size': 16,
    'num_epochs': 100,
    'patience': 50,
}
```

## 文件结构

```
calibration-BCI/
├── run_model_training.sh          # 模型训练工作流
├── run_calibration.sh             # 校准工作流
├── config_forward_comprehensive.py # 配置文件
│
├── 数据准备
│   ├── extract_forcing_data.py
│   └── noahmp/TBL_generator/
│       ├── generate_samples.py
│       ├── noahmp_apply_samples.py
│       └── value_bounds.csv
│
├── 训练
│   ├── 01_data_preprocessing_forward_comprehensive.py
│   └── 02_train_forward_comprehensive.py
│
├── 校准验证
│   ├── 06_calibration_applying_emulator_multiple_runs.py
│   ├── 07_calibration_validation_enhanced.py
│   └── src/convert_calibrated_params.py
│
└── 数据目录
    ├── data/raw/forcing/          # forcing数据
    ├── data/raw/sim_results/      # Noah-MP输出
    ├── data/obs/                  # 观测数据
    ├── calibration_results/       # 校准结果
    └── validation_results/        # 验证结果
```

## 常见问题

**Q: 预处理时提示timesteps不匹配**
```bash
# 确保forcing数据覆盖Noah-MP输出的时间范围
python3 extract_forcing_data.py --daily \
    --start_date 2015-07-30 --end_date 2017-07-30 \
    --output data/raw/forcing/forcing_panama_daily_full.nc
```

**Q: 训练时内存不足**
```python
# 在config中减小batch_size或使用daily分辨率
TRAINING_CONFIG = {'batch_size': 8}
TEMPORAL_RESOLUTION = 'daily'
```

**Q: 校准后SATDK值异常**
```
# 检查value_bounds.csv中是否同时有SATDK和SATDK(log)两行
# 校准使用log bounds，转换时自动还原为物理值
```

---
**Last Updated:** 2024-12-18
