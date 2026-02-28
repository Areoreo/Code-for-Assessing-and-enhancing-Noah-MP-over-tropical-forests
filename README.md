# LSTM Emulator for Noah-MP Calibration

An LSTM neural network-based rapid parameter calibration system for the Noah-MP land surface model.

## Workflow Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                     run_model_training.sh                           │
│  1. Generate parameter samples (LHS)                                │
│  2. Extract forcing data (full/calibration/validation periods)      │
│  3. Run Noah-MP (1000 parameter sets × full period)                 │
│  4. Preprocess data (calibration period)                            │
│  5. Train LSTM emulator                                             │
└─────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────┐
│                       run_calibration.sh                            │
│  1. Calibrate parameters using emulator (calibration period)        │
│  2. Run Noah-MP validation with calibrated parameters (full period) │
│  3. Analyze and plot results (calibration/validation periods)       │
└─────────────────────────────────────────────────────────────────────┘
```

## Quick Start

```bash
cd /home/petrichor/ymwang/snap/Emulator-based_calibration/calibration-BCI

# Full model training workflow
bash run_model_training.sh --samples 1000 --parallel 4

# Full calibration workflow
bash run_calibration.sh --num_calibration 10 --max_iter 100
```

## Time Range Configuration

Defined in `config_forward_comprehensive.py`:

| Period      | Range                   | Purpose                                  |
| ----------- | ----------------------- | ---------------------------------------- |
| FULL        | 2015-07-30 ~ 2017-07-30 | Noah-MP runs, validation                 |
| CALIBRATION | 2015-07-30 ~ 2016-07-29 | Emulator training, parameter calibration |
| VALIDATION  | 2016-07-30 ~ 2017-07-30 | Independent validation                   |

## Detailed Steps

### Model Training (`run_model_training.sh`)

```bash
# Full run
bash run_model_training.sh

# Skip completed steps
bash run_model_training.sh --skip-param-gen --skip-forcing
bash run_model_training.sh --skip-noahmp
bash run_model_training.sh --skip-preprocess --skip-train

# Optional parameters
--samples N      # Number of parameter samples (default: 1000)
--parallel N     # Number of parallel jobs (default: 4)
```

**Output files:**

- `data/raw/param/noahmp_param_sets.txt` - Parameter samples
- `data/raw/forcing/forcing_panama_*.nc` - Forcing data
- `data/raw/sim_results/sample_*/` - Noah-MP outputs
- `data/processed_data_forward_comprehensive.pkl` - Training data
- `results_forward_comprehensive/*/` - Trained model

### Parameter Calibration (`run_calibration.sh`)

```bash
# Full run
bash run_calibration.sh

# Specify model directory
bash run_calibration.sh --model_dir results_forward_comprehensive/AttentionLSTM_xxx/

# Adjust calibration parameters
bash run_calibration.sh --num_calibration 20 --max_iter 200

# Skip steps
bash run_calibration.sh --skip-calibration  # Use existing calibration results
bash run_calibration.sh --skip-noahmp       # Skip validation runs
```

**Output files:**

- `calibration_results/*/calibration_1/` - Calibration results
  - `calibrated_parameters.csv` - Calibrated parameters
  - `calibration_result.json` - Detailed metrics
- `validation_results/*/` - Validation results
  - `validation_metrics_by_period.csv` - Metrics by period
  - `timeseries_*.png/pdf` - Time series comparison plots
  - `scatter_*.png/pdf` - Scatter plots
  - `metrics_*.png/pdf` - Metrics comparison plots

## SATDK Parameter Handling

The SATDK parameter uses different forms at different stages:

| Stage                         | Value Form  | Description                               |
| ----------------------------- | ----------- | ----------------------------------------- |
| Parameter sample generation   | Raw value   | `[8.9e-06, 5e-04]`                        |
| TBL file generation           | Raw value   | Written directly to Noah-MP configuration |
| Emulator training             | log10 value | Automatically converted `→ [-5.05, -3.3]` |
| Parameter calibration         | log10 value | Uses `SATDK(log)` bounds                  |
| Calibration result conversion | log→raw     | `10^x` conversion back to physical value  |

**value_bounds.csv format:**

```csv
variable,Lower bound,Upper bound
SATDK,8.90E-06,5.00E-04
SATDK(log),-5.05,-3.3
```

## Running Individual Steps

### Generate Parameter Samples

```bash
cd noahmp/TBL_generator
python3 generate_samples.py
```

### Extract Forcing Data

```bash
# Full period
python3 extract_forcing_data.py --daily \
    --start_date 2015-07-30 --end_date 2017-07-30 \
    --output data/raw/forcing/forcing_panama_daily_full.nc

# Calibration period
python3 extract_forcing_data.py --daily \
    --start_date 2015-07-30 --end_date 2016-07-29 \
    --output data/raw/forcing/forcing_panama_daily_calibration.nc
```

### Data Preprocessing

```bash
# Full period
python3 01_data_preprocessing_forward_comprehensive.py

# Calibration period (for emulator training)
python3 01_data_preprocessing_forward_comprehensive.py --calibration
```

### Train Emulator

```bash
python3 02_train_forward_comprehensive.py
```

### Parameter Calibration

```bash
python3 05_calibration_applying_emulator_multiple_runs.py \
    --model_dir results_forward_comprehensive/AttentionLSTM_xxx/ \
    --forcing data/raw/forcing/forcing_panama_daily_calibration.nc \
    --obs data/obs/Panama_BCI_obs_2015-07-30_2016-07-29.csv \
    --bounds value_bounds.csv \
    --num_calibration 10 \
    --max_iter 100
```

## Variable Description

### Forcing Variables (8)

| Variable | Description                         |
| -------- | ----------------------------------- |
| T2D      | 2m air temperature (K)              |
| Q2D      | 2m specific humidity (kg/kg)        |
| PSFC     | Surface pressure (Pa)               |
| U2D, V2D | 2m wind speed components (m/s)      |
| LWDOWN   | Downward longwave radiation (W/m²)  |
| SWDOWN   | Downward shortwave radiation (W/m²) |
| RAINRATE | Precipitation rate (mm/s)           |

### Target Variables (29)

- **Energy balance** (5): FSA, FIRA, HFX, LH, GRDFLX
- **Water fluxes** (5): ECAN, ETRAN, EDIR, UGDRNOFF_RATE, SFCRNOFF_RATE
- **Water storage** (5): SOIL_M (L1-L4), CANLIQ
- **Temperature** (5): SOIL_T (L1-L2), TG, TV, TRAD
- **Energy components** (9): SAV, SAG, IRC, IRG, SHC, SHG, EVC, EVG, GHV

### Calibration Parameters (9)

| Parameter | Description                      | Range          |
| --------- | -------------------------------- | -------------- |
| VCMX25    | Maximum carboxylation rate       | [30, 120]      |
| HVT       | Canopy top height                | [9, 55]        |
| HVB       | Canopy bottom height             | [0.1, 15]      |
| CWPVT     | Canopy wind parameter            | [0.15, 5.35]   |
| Z0MVT     | Momentum roughness length        | [0.3, 2]       |
| WLTSMC    | Wilting point soil moisture      | [0.02, 0.26]   |
| REFSMC    | Reference soil moisture          | [0.15, 0.45]   |
| MAXSMC    | Saturated soil moisture          | [0.45, 0.75]   |
| SATDK     | Saturated hydraulic conductivity | [8.9e-6, 5e-4] |

## Model Configuration

Edit `config_forward_comprehensive.py`:

```python
TEMPORAL_RESOLUTION = 'daily'  # 'daily' or '30min'

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

## File Structure

```
project/
├── run_model_training.sh          # Model training workflow
├── run_calibration.sh             # Calibration workflow
├── config_forward_comprehensive.py # Configuration file
│
├── Data Preparation
│   ├── extract_forcing_data.py
│   └── noahmp/TBL_generator/
│       ├── generate_samples.py
│       ├── noahmp_apply_samples.py
│       └── value_bounds.csv
│
├── Training
│   ├── 01_data_preprocessing_forward_comprehensive.py
│   └── 02_train_forward_comprehensive.py
│
├── Validation & Calibration
│   ├── 05_calibration_applying_emulator_multiple_runs.py
│   ├── 06_calibration_validation.py
│   ├── 06_calibration_validation.sh
│   ├── 07_importance_analysis.py
│   ├── 07_importance_analysis.sh
│   └── src/
│       ├── convert_calibrated_params.py
│       ├── convert_pso_obs.py
│       ├── plot_validation_results.py
│       └── plot_validation_scatter.py
│
└── Data Directories
    ├── data/raw/forcing/          # Forcing data
    ├── data/raw/sim_results/      # Noah-MP outputs
    ├── data/obs/                  # Observation data
    ├── calibration_results/       # Calibration results
    └── validation_results/        # Validation results
```

## FAQ

**Q: Timestep mismatch during preprocessing**

```bash
# Ensure forcing data covers the Noah-MP output time range
python3 extract_forcing_data.py --daily \
    --start_date 2015-07-30 --end_date 2017-07-30 \
    --output data/raw/forcing/forcing_panama_daily_full.nc
```

**Q: Out of memory during training**

```python
# Reduce batch_size in config or use daily resolution
TRAINING_CONFIG = {'batch_size': 8}
TEMPORAL_RESOLUTION = 'daily'
```

---

**Last Updated:** 2025-12-18
