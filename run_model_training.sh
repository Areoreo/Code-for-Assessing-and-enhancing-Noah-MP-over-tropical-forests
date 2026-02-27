#!/bin/bash
####################################################################################
# Model Training Workflow Script
#
# This script orchestrates the complete model training workflow:
# 1. Generate parameter sets using Latin Hypercube Sampling (with SATDK in log form)
# 2. Extract forcing data for different time ranges (full, calibration, validation)
# 3. Run Noah-MP with each parameter set for FULL time range
# 4. Preprocess simulation outputs for emulator training (using CALIBRATION time range)
# 5. Train the LSTM emulator (using CALIBRATION time range data only)
#
# Time ranges are defined in config_forward_comprehensive.py:
#   - TIME_RANGE_FULL: Full data range for Noah-MP runs (e.g., 2015-07-30 to 2017-07-29)
#   - TIME_RANGE_CALIBRATION: For emulator training (e.g., 2015-07-30 to 2016-07-29)
#   - TIME_RANGE_VALIDATION: For later validation (e.g., 2016-07-30 to 2017-07-29)
#
# Usage: bash run_model_training.sh [OPTIONS]
####################################################################################

set -e

# =============================================================================
# Configuration
# =============================================================================
BASE_DIR="/home/petrichor/ymwang/snap/Emulator-based_calibration/calibration-PSO"
NOAHMP_BASE="/home/petrichor/ymwang/snap/Noah-mp"
FORCING_SOURCE="${NOAHMP_BASE}/data/PSO_single_point"
TBL_GENERATOR_DIR="${BASE_DIR}/noahmp/TBL_generator"
POINT_RUN_DIR="${BASE_DIR}/noahmp/point_run"

# Default settings
N_SAMPLES=1000
N_PARALLEL=4
SKIP_PARAM_GEN=false
SKIP_FORCING=false
SKIP_NOAHMP=false
SKIP_PREPROCESS=false
SKIP_TRAIN=false

# Timestamp for this run
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="${BASE_DIR}/logs/${TIMESTAMP}"

# =============================================================================
# Helper Functions
# =============================================================================
print_header() {
    echo ""
    echo "================================================================================="
    echo " $1"
    echo "================================================================================="
}

print_info() {
    echo -e "\033[1;34m[INFO]\033[0m $1"
}

print_success() {
    echo -e "\033[1;32m[SUCCESS]\033[0m $1"
}

print_error() {
    echo -e "\033[1;31m[ERROR]\033[0m $1"
}

print_warning() {
    echo -e "\033[1;33m[WARNING]\033[0m $1"
}

# Function to extract time ranges from config
get_time_config() {
    python3 << 'PYEOF'
import sys
sys.path.insert(0, '/home/petrichor/ymwang/snap/Emulator-based_calibration/calibration-PSO')
import config_forward_comprehensive as config

print(f"FULL_START={config.TIME_RANGE_FULL['start']}")
print(f"FULL_END={config.TIME_RANGE_FULL['end']}")
print(f"CAL_START={config.TIME_RANGE_CALIBRATION['start']}")
print(f"CAL_END={config.TIME_RANGE_CALIBRATION['end']}")
print(f"VAL_START={config.TIME_RANGE_VALIDATION['start']}")
print(f"VAL_END={config.TIME_RANGE_VALIDATION['end']}")
PYEOF
}

# =============================================================================
# Parse Arguments
# =============================================================================
while [[ $# -gt 0 ]]; do
    case $1 in
        --skip-param-gen)
            SKIP_PARAM_GEN=true
            shift
            ;;
        --skip-forcing)
            SKIP_FORCING=true
            shift
            ;;
        --skip-noahmp)
            SKIP_NOAHMP=true
            shift
            ;;
        --skip-preprocess)
            SKIP_PREPROCESS=true
            shift
            ;;
        --skip-train)
            SKIP_TRAIN=true
            shift
            ;;
        --parallel)
            N_PARALLEL="$2"
            shift 2
            ;;
        --samples)
            N_SAMPLES="$2"
            shift 2
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Model Training Workflow - Train LSTM emulator for Noah-MP"
            echo ""
            echo "Options:"
            echo "  --skip-param-gen    Skip parameter generation"
            echo "  --skip-forcing      Skip forcing data extraction"
            echo "  --skip-noahmp       Skip Noah-MP runs"
            echo "  --skip-preprocess   Skip data preprocessing"
            echo "  --skip-train        Skip emulator training"
            echo "  --parallel N        Parallel runs (default: 4)"
            echo "  --samples N         Number of samples (default: 1000)"
            echo ""
            echo "Time ranges are configured in config_forward_comprehensive.py"
            exit 0
            ;;
        *)
            print_error "Unknown option: $1"
            exit 1
            ;;
    esac
done

# =============================================================================
# Initialize
# =============================================================================
print_header "MODEL TRAINING WORKFLOW"
print_info "Timestamp: ${TIMESTAMP}"
print_info "Base directory: ${BASE_DIR}"
print_info "Number of samples: ${N_SAMPLES}"
print_info "Parallel runs: ${N_PARALLEL}"

mkdir -p "${LOG_DIR}"
print_info "Log directory: ${LOG_DIR}"

cd "${BASE_DIR}"

# Get time configuration
print_info "Loading time configuration from config_forward_comprehensive.py..."
eval "$(get_time_config)"

echo ""
print_info "Time Configuration:"
print_info "  Full range:        ${FULL_START} to ${FULL_END}"
print_info "  Calibration range: ${CAL_START} to ${CAL_END}"
print_info "  Validation range:  ${VAL_START} to ${VAL_END}"
echo ""

# =============================================================================
# Step 1: Generate Parameter Sets
# =============================================================================
if [ "${SKIP_PARAM_GEN}" = false ]; then
    print_header "Step 1: Generate ${N_SAMPLES} Parameter Sets"
    
    cd "${TBL_GENERATOR_DIR}"
    
    # Update n_samples in generate_samples.py
    sed -i "s/n_samples = .*/n_samples = ${N_SAMPLES}/" generate_samples.py
    
    print_info "Generating parameter sets with Latin Hypercube Sampling..."
    print_info "Note: SATDK is stored in log10 form (will be converted during TBL generation)"
    
    python3 generate_samples.py 2>&1 | tee "${LOG_DIR}/01_param_generation.log"
    
    # Copy to data directory
    cp noahmp_1000samples_new.txt "${BASE_DIR}/data/raw/param/noahmp_param_sets.txt"
    
    print_success "Generated ${N_SAMPLES} parameter sets"
    print_info "Parameters saved to: ${BASE_DIR}/data/raw/param/noahmp_param_sets.txt"
    
    cd "${BASE_DIR}"
else
    print_info "Skipping parameter generation"
fi

# =============================================================================
# Step 2: Extract Forcing Data for Different Time Ranges
# =============================================================================
if [ "${SKIP_FORCING}" = false ]; then
    print_header "Step 2: Extract Forcing Data for Different Time Ranges"
    
    # Extract FULL time range - 30min resolution (for Noah-MP runs)
    print_info "Extracting 30-min forcing data for FULL time range..."
    python3 extract_forcing_data.py \
        --forcing_dir "${FORCING_SOURCE}" \
        --output data/raw/forcing/forcing_PSO_30min.nc \
        --start_date "${FULL_START}" \
        --end_date "${FULL_END}" \
        2>&1 | tee "${LOG_DIR}/02_forcing_30min_full.log"
    
    # Extract FULL time range - daily resolution (for LSTM)
    print_info "Extracting daily forcing data for FULL time range..."
    python3 extract_forcing_data.py \
        --forcing_dir "${FORCING_SOURCE}" \
        --output data/raw/forcing/forcing_PSO_daily_full.nc \
        --daily \
        --start_date "${FULL_START}" \
        --end_date "${FULL_END}" \
        2>&1 | tee "${LOG_DIR}/02_forcing_daily_full.log"
    
    # Extract CALIBRATION time range - daily resolution (for LSTM training)
    print_info "Extracting daily forcing data for CALIBRATION time range..."
    python3 extract_forcing_data.py \
        --forcing_dir "${FORCING_SOURCE}" \
        --output data/raw/forcing/forcing_PSO_daily_calibration.nc \
        --daily \
        --start_date "${CAL_START}" \
        --end_date "${CAL_END}" \
        2>&1 | tee "${LOG_DIR}/02_forcing_daily_calibration.log"
    
    # Extract VALIDATION time range - daily resolution (for later use)
    print_info "Extracting daily forcing data for VALIDATION time range..."
    python3 extract_forcing_data.py \
        --forcing_dir "${FORCING_SOURCE}" \
        --output data/raw/forcing/forcing_PSO_daily_validation.nc \
        --daily \
        --start_date "${VAL_START}" \
        --end_date "${VAL_END}" \
        2>&1 | tee "${LOG_DIR}/02_forcing_daily_validation.log"
    
    # Create symbolic link for backward compatibility
    ln -sf forcing_PSO_daily_calibration.nc data/raw/forcing/forcing_PSO_daily.nc
    
    print_success "Forcing data extracted for all time ranges"
else
    print_info "Skipping forcing data extraction"
fi

# =============================================================================
# Step 3: Run Noah-MP for All Parameter Sets (FULL Time Range)
# =============================================================================
if [ "${SKIP_NOAHMP}" = false ]; then
    print_header "Step 3: Run Noah-MP Simulations (FULL Time Range: ${FULL_START} to ${FULL_END})"
    
    print_warning "Note: TBL generator will convert SATDK from log10 to real values"
    
    bash run_noahmp_batch.sh \
        --samples ${N_SAMPLES} \
        --parallel ${N_PARALLEL} \
        2>&1 | tee "${LOG_DIR}/03_noahmp_runs.log"
    
    print_success "Completed Noah-MP simulations for ${N_SAMPLES} parameter sets"
else
    print_info "Skipping Noah-MP simulations"
fi

# =============================================================================
# Step 4: Data Preprocessing (Using CALIBRATION Time Range)
# =============================================================================
if [ "${SKIP_PREPROCESS}" = false ]; then
    print_header "Step 4: Data Preprocessing (CALIBRATION Time Range: ${CAL_START} to ${CAL_END})"
    
    print_info "Preprocessing simulation outputs for emulator training..."
    print_info "Using calibration time range for training data"
    
    # Update config to use calibration forcing file
    python3 01_data_preprocessing_forward_comprehensive.py --calibration \
        2>&1 | tee "${LOG_DIR}/04_preprocessing.log"
    
    print_success "Data preprocessing completed"
else
    print_info "Skipping data preprocessing"
fi

# =============================================================================
# Step 5: Train Emulator (Using CALIBRATION Data)
# =============================================================================
if [ "${SKIP_TRAIN}" = false ]; then
    print_header "Step 5: Train LSTM Emulator (Using CALIBRATION Data)"
    
    print_info "Training emulator using calibration time range data..."
    
    python3 02_train_forward_comprehensive.py \
        2>&1 | tee "${LOG_DIR}/05_training.log"
    
    MODEL_DIR=$(ls -td results_forward_comprehensive/*/ | head -1)
    print_success "Emulator trained successfully!"
    print_info "Model saved to: ${MODEL_DIR}"
    echo "${MODEL_DIR}" > "${LOG_DIR}/model_dir.txt"
else
    print_info "Skipping emulator training"
    MODEL_DIR=$(ls -td results_forward_comprehensive/*/ | head -1)
    print_info "Using existing model: ${MODEL_DIR}"
fi

# =============================================================================
# Summary
# =============================================================================
print_header "MODEL TRAINING WORKFLOW COMPLETED"
echo ""
print_info "Summary:"
print_info "  Timestamp:          ${TIMESTAMP}"
print_info "  Log directory:      ${LOG_DIR}"
print_info "  Parameter sets:     ${N_SAMPLES}"
print_info "  Full time range:    ${FULL_START} to ${FULL_END}"
print_info "  Training range:     ${CAL_START} to ${CAL_END}"
print_info "  Model directory:    ${MODEL_DIR}"
echo ""
print_info "Generated files:"
print_info "  - data/raw/param/noahmp_param_sets.txt"
print_info "  - data/raw/forcing/forcing_PSO_30min.nc"
print_info "  - data/raw/forcing/forcing_PSO_daily_full.nc"
print_info "  - data/raw/forcing/forcing_PSO_daily_calibration.nc"
print_info "  - data/raw/forcing/forcing_PSO_daily_validation.nc"
print_info "  - data/processed_data_forward_comprehensive.pkl"
print_info "  - ${MODEL_DIR}"
echo ""
print_success "Model training workflow completed!"
print_info "Next step: Run calibration using run_calibration.sh"
