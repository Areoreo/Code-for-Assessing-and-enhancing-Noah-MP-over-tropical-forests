#!/bin/bash
####################################################################################
# Calibration Workflow Script (V2 - Loop All Calibrations)
#
# Improvements:
#   1. New --calibration_dir option to specify existing calibration results
#   2. Loop through ALL calibration results for validation (not just one)
#
# Usage: bash run_calibration.sh [OPTIONS]
####################################################################################

set -e

# =============================================================================
# Configuration
# =============================================================================
BASE_DIR="/home/petrichor/ymwang/snap/Emulator-based_calibration/calibration-PSO"
NOAHMP_DIR="${BASE_DIR}/noahmp/point_run"
TBL_GENERATOR_DIR="${BASE_DIR}/noahmp/TBL_generator"
PARAM_DIR="${BASE_DIR}/data/raw/param"
OBS_DIR="${BASE_DIR}/data/obs"
VALIDATION_RESULTS_DIR="${BASE_DIR}/validation_results"
SRC_DIR="${BASE_DIR}/src"

# Default settings
NUM_CALIBRATION=10
MAX_ITER=100
POPSIZE=15
SKIP_CALIBRATION=false
SKIP_NOAHMP=false
SKIP_ANALYSIS=false

# Model directory (auto-detect latest)
MODEL_DIR=""

# [NEW] User-specified calibration directory (for --skip-calibration)
USER_CALIBRATION_DIR=""

# Timestamp for this run
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="${BASE_DIR}/logs/calibration_${TIMESTAMP}"

# Variables to validate
VARIABLES="SOIL_M LH HFX"

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
        --model_dir)
            MODEL_DIR="$2"
            shift 2
            ;;
        --num_calibration)
            NUM_CALIBRATION="$2"
            shift 2
            ;;
        --max_iter)
            MAX_ITER="$2"
            shift 2
            ;;
        --popsize)
            POPSIZE="$2"
            shift 2
            ;;
        --skip-calibration)
            SKIP_CALIBRATION=true
            shift
            ;;
        # [NEW] Option to specify calibration results directory
        --calibration_dir)
            USER_CALIBRATION_DIR="$2"
            shift 2
            ;;
        --skip-noahmp)
            SKIP_NOAHMP=true
            shift
            ;;
        --skip-analysis)
            SKIP_ANALYSIS=true
            shift
            ;;
        --variables)
            VARIABLES="$2"
            shift 2
            ;;
        --help|-h)
            cat << EOF
Usage: $0 [OPTIONS]

Calibration Workflow - Calibrate Noah-MP parameters using LSTM emulator

Options:
  --model_dir DIR       Path to trained emulator model directory
                        (default: auto-detect latest)
  --num_calibration N   Number of independent calibration runs (default: 10)
  --max_iter N          Maximum optimization iterations (default: 100)
  --popsize N           Population size for differential evolution (default: 15)
  --variables VARS      Variables to validate (default: "SOIL_M LH HFX")
  --skip-calibration    Skip calibration step (use existing results)
  --calibration_dir DIR [NEW] Specify calibration results directory 
                        (used with --skip-calibration, e.g., calibration_results/20250101_120000)
  --skip-noahmp         Skip Noah-MP validation runs
  --skip-analysis       Skip analysis and plotting
  -h, --help            Show this help message

Time ranges are configured in config_forward_comprehensive.py

Example:
  # Run full calibration workflow
  bash run_calibration.sh --num_calibration 20 --max_iter 200

  # Use specific existing calibration directory
  bash run_calibration.sh --skip-calibration --calibration_dir calibration_results/20250101_120000

  # Use latest calibration (auto-detect)
  bash run_calibration.sh --skip-calibration

EOF
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
print_header "CALIBRATION WORKFLOW"
print_info "Timestamp: ${TIMESTAMP}"
print_info "Base directory: ${BASE_DIR}"

mkdir -p "${LOG_DIR}"
print_info "Log directory: ${LOG_DIR}"

cd "${BASE_DIR}"

# Auto-detect model directory if not specified
if [ -z "${MODEL_DIR}" ]; then
    MODEL_DIR=$(ls -td results_forward_comprehensive/*/ 2>/dev/null | head -1)
    if [ -z "${MODEL_DIR}" ]; then
        print_error "No trained model found. Run run_model_training.sh first."
        exit 1
    fi
fi
print_info "Model directory: ${MODEL_DIR}"

# Get time configuration
print_info "Loading time configuration from config_forward_comprehensive.py..."
eval "$(get_time_config)"

echo ""
print_info "Time Configuration:"
print_info "  Full range:        ${FULL_START} to ${FULL_END}"
print_info "  Calibration range: ${CAL_START} to ${CAL_END}"
print_info "  Validation range:  ${VAL_START} to ${VAL_END}"
echo ""

# Define observation files (PSO specific)
OBS_CALIBRATION="${OBS_DIR}/Malaysia_PSO_obs_${CAL_START}_${CAL_END}.csv"
OBS_FULL="${OBS_DIR}/PSO_obs_all_combined_30min.csv"

# Check observation files
if [ ! -f "${OBS_CALIBRATION}" ]; then
    print_warning "Calibration observation file not found: ${OBS_CALIBRATION}"
    print_info "Trying alternative observation file..."
    OBS_CALIBRATION="${OBS_DIR}/Malaysia_PSO_obs_2003-01-01_2004-01-01.csv"
fi

if [ ! -f "${OBS_CALIBRATION}" ]; then
    print_error "Observation file not found. Please check ${OBS_DIR}"
    exit 1
fi

print_info "Using observation file: ${OBS_CALIBRATION}"

# Define calibration output directory
CALIBRATION_OUTPUT="${BASE_DIR}/calibration_results/${TIMESTAMP}"

# =============================================================================
# Step 1: Run Calibration using Emulator (CALIBRATION Time Range)
# =============================================================================
if [ "${SKIP_CALIBRATION}" = false ]; then
    print_header "Step 1: Parameter Calibration (${CAL_START} to ${CAL_END})"
    
    print_info "Running ${NUM_CALIBRATION} independent calibrations..."
    print_info "Using differential evolution optimizer"
    print_info "Max iterations: ${MAX_ITER}, Population size: ${POPSIZE}"
    
    python3 05_calibration_applying_emulator_multiple_runs.py \
        --model_dir "${MODEL_DIR}" \
        --forcing data/raw/forcing/forcing_PSO_daily_calibration.nc \
        --obs "${OBS_CALIBRATION}" \
        --bounds value_bounds.csv \
        --num_calibration ${NUM_CALIBRATION} \
        --max_iter ${MAX_ITER} \
        --popsize ${POPSIZE} \
        --output "${CALIBRATION_OUTPUT}" \
        --weights 1 1.5 2.5 \
        --loss_type nrmse
        2>&1 | tee "${LOG_DIR}/01_calibration.log"
    
    print_success "Calibration completed!"
    print_info "Results saved to: ${CALIBRATION_OUTPUT}"
else
    print_info "Skipping calibration step"
    
    # [IMPROVED] Use user-specified directory or auto-detect latest
    if [ -n "${USER_CALIBRATION_DIR}" ]; then
        # User specified a directory
        if [[ "${USER_CALIBRATION_DIR}" == /* ]]; then
            # Absolute path
            CALIBRATION_OUTPUT="${USER_CALIBRATION_DIR}"
        else
            # Relative path - assume under BASE_DIR
            CALIBRATION_OUTPUT="${BASE_DIR}/${USER_CALIBRATION_DIR}"
        fi
        
        if [ ! -d "${CALIBRATION_OUTPUT}" ]; then
            print_error "Specified calibration directory does not exist: ${CALIBRATION_OUTPUT}"
            exit 1
        fi
        print_info "Using user-specified calibration directory: ${CALIBRATION_OUTPUT}"
    else
        # Auto-detect latest
        CALIBRATION_OUTPUT=$(ls -td ${BASE_DIR}/calibration_results/*/ 2>/dev/null | head -1)
        if [ -z "${CALIBRATION_OUTPUT}" ]; then
            print_error "No calibration results found. Run calibration first or specify --calibration_dir"
            exit 1
        fi
        print_info "Auto-detected latest calibration: ${CALIBRATION_OUTPUT}"
    fi
fi

# Remove trailing slash if present
CALIBRATION_OUTPUT="${CALIBRATION_OUTPUT%/}"

# [NEW] Find ALL calibration subdirectories
CALIBRATION_SUBDIRS=($(ls -d ${CALIBRATION_OUTPUT}/calibration_* 2>/dev/null | sort -V))
NUM_CAL_RESULTS=${#CALIBRATION_SUBDIRS[@]}

if [ ${NUM_CAL_RESULTS} -eq 0 ]; then
    print_error "No calibration_* subdirectories found in ${CALIBRATION_OUTPUT}"
    exit 1
fi

print_info "Found ${NUM_CAL_RESULTS} calibration result(s) to validate"

# =============================================================================
# Step 2: Validation - Run Noah-MP with Different Parameter Sets (FULL Time Range)
# [IMPROVED] Loop through ALL calibration results
# =============================================================================
if [ "${SKIP_NOAHMP}" = false ]; then
    print_header "Step 2: Validation Runs (${FULL_START} to ${FULL_END})"
    
    # Function to run single Noah-MP simulation
    run_noahmp_validation() {
        local tbl_file="$1"
        local run_name="$2"
        local output_base_dir="$3"
        local run_dir="${output_base_dir}/run_${run_name}"
        
        echo "[INFO] Running Noah-MP with ${run_name} parameters..."
        
        mkdir -p "${run_dir}/output"
        cp "${NOAHMP_DIR}/hrldas.exe" "${run_dir}/"
        cp "${NOAHMP_DIR}/namelist.hrldas" "${run_dir}/"
        cp "${tbl_file}" "${run_dir}/NoahmpTable.TBL"
        
        # Create symlink to forcing data (PSO specific)
        ln -sf "/home/petrichor/ymwang/snap/Noah-mp/data/PSO_single_point" "${run_dir}/forcing"
        
        cd "${run_dir}"
        sed -i "s|INDIR.*=.*|INDIR = './forcing'|" namelist.hrldas
        sed -i "s|OUTDIR.*=.*|OUTDIR = './output'|" namelist.hrldas
        
        export LD_LIBRARY_PATH=/home/petrichor/ymwang/.local/lib:/usr/local/cuda-12.1/lib64:$LD_LIBRARY_PATH
        ./hrldas.exe > run.log 2>&1
        
        if ls output/*.LDASOUT_DOMAIN1 1> /dev/null 2>&1; then
            echo "[SUCCESS] ${run_name} simulation completed"
        else
            echo "[ERROR] ${run_name} simulation failed. Check ${run_dir}/run.log"
        fi
        
        cd "${BASE_DIR}"
    }
    
    # [NEW] Loop through each calibration result
    for cal_subdir in "${CALIBRATION_SUBDIRS[@]}"; do
        CAL_NAME=$(basename "${cal_subdir}")
        TIMESTAMP_DIR=$(basename "${CALIBRATION_OUTPUT}")
        CALIBRATION_ID="${TIMESTAMP_DIR}/${CAL_NAME}"
        
        print_info "=========================================="
        print_info "Processing: ${CALIBRATION_ID}"
        print_info "=========================================="
        
        VALIDATION_OUTPUT_DIR="${VALIDATION_RESULTS_DIR}/${CALIBRATION_ID}"
        mkdir -p "${VALIDATION_OUTPUT_DIR}"
        
        # Step 2a: Convert calibrated parameters to TBL format
        print_info "Converting calibrated parameters to TBL format..."
        EMULATOR_PARAM_FILE="${VALIDATION_OUTPUT_DIR}/emulator_calibrated_params.txt"
        
        python3 "${SRC_DIR}/convert_calibrated_params.py" \
            --input "${cal_subdir}/calibrated_parameters.csv" \
            --output "${EMULATOR_PARAM_FILE}" \
            2>&1 | tee -a "${LOG_DIR}/02_validation_${CAL_NAME}.log"
        
        # Step 2b: Generate TBL files
        print_info "Generating TBL files..."
        cd "${TBL_GENERATOR_DIR}"
        
        # Emulator-calibrated
        python3 noahmp_apply_samples.py \
            --samples "${EMULATOR_PARAM_FILE}" \
            --base_table NoahmpTable.TBL \
            --out_root "${VALIDATION_OUTPUT_DIR}/emulator_tbl" \
            --n_rows 1 \
            --verbose
        
        EMULATOR_TBL="${VALIDATION_OUTPUT_DIR}/emulator_tbl/NoahmpTable_4emu_1/NoahmpTable.TBL"
        
        # Expert-calibrated
        python3 noahmp_apply_samples.py \
            --samples "${PARAM_DIR}/expert_calibration.txt" \
            --base_table NoahmpTable.TBL \
            --out_root "${VALIDATION_OUTPUT_DIR}/expert_tbl" \
            --n_rows 1
        
        EXPERT_TBL="${VALIDATION_OUTPUT_DIR}/expert_tbl/NoahmpTable_4emu_1/NoahmpTable.TBL"
        
        # Default parameters
        python3 noahmp_apply_samples.py \
            --samples "${PARAM_DIR}/default_param.txt" \
            --base_table NoahmpTable.TBL \
            --out_root "${VALIDATION_OUTPUT_DIR}/default_tbl" \
            --n_rows 1
        
        DEFAULT_TBL="${VALIDATION_OUTPUT_DIR}/default_tbl/NoahmpTable_4emu_1/NoahmpTable.TBL"
        
        cd "${BASE_DIR}"
        
        # Step 2c: Run Noah-MP with each parameter set
        print_info "Running Noah-MP simulations for ${CAL_NAME}..."
        
        run_noahmp_validation "${EMULATOR_TBL}" "emulator" "${VALIDATION_OUTPUT_DIR}" 2>&1 | tee -a "${LOG_DIR}/02_validation_${CAL_NAME}.log"
        run_noahmp_validation "${EXPERT_TBL}" "expert" "${VALIDATION_OUTPUT_DIR}" 2>&1 | tee -a "${LOG_DIR}/02_validation_${CAL_NAME}.log"
        run_noahmp_validation "${DEFAULT_TBL}" "default" "${VALIDATION_OUTPUT_DIR}" 2>&1 | tee -a "${LOG_DIR}/02_validation_${CAL_NAME}.log"
        
        # Step 2d: Parse Noah-MP outputs
        print_info "Parsing Noah-MP outputs to CSV..."
        
        for run_type in emulator expert default; do
            output_file=$(ls ${VALIDATION_OUTPUT_DIR}/run_${run_type}/output/*.LDASOUT_DOMAIN1 2>/dev/null | head -1)
            if [ -n "${output_file}" ]; then
                python3 "${SRC_DIR}/parse_noahmp_outputs.py" \
                    --input "${output_file}" \
                    --output "${VALIDATION_OUTPUT_DIR}/${run_type}_output.csv"
            fi
        done
        
        print_success "Validation completed for ${CAL_NAME}"
    done
    
    print_success "All validation runs completed!"
else
    print_info "Skipping Noah-MP validation runs"
fi

# =============================================================================
# Step 3: Analysis and Plotting
# [IMPROVED] Analyze ALL calibration results
# =============================================================================
if [ "${SKIP_ANALYSIS}" = false ]; then
    print_header "Step 3: Analysis and Plotting"
    
    # Loop through each calibration result for analysis
    for cal_subdir in "${CALIBRATION_SUBDIRS[@]}"; do
        CAL_NAME=$(basename "${cal_subdir}")
        TIMESTAMP_DIR=$(basename "${CALIBRATION_OUTPUT}")
        CALIBRATION_ID="${TIMESTAMP_DIR}/${CAL_NAME}"
        VALIDATION_OUTPUT_DIR="${VALIDATION_RESULTS_DIR}/${CALIBRATION_ID}"
        
        print_info "Analyzing: ${CALIBRATION_ID}"
        
        # Check if validation outputs exist
        if [ ! -f "${VALIDATION_OUTPUT_DIR}/emulator_output.csv" ]; then
            print_warning "Skipping ${CAL_NAME}: validation outputs not found"
            continue
        fi
        
        # Run enhanced validation analysis (PSO specific observation file)
        python3 06_calibration_validation.py \
            --obs "${OBS_DIR}/PSO_obs_all_combined_30min.csv" \
            --emulator_output "${VALIDATION_OUTPUT_DIR}/emulator_output.csv" \
            --expert_output "${VALIDATION_OUTPUT_DIR}/expert_output.csv" \
            --default_output "${VALIDATION_OUTPUT_DIR}/default_output.csv" \
            --output_dir "${VALIDATION_OUTPUT_DIR}" \
            --variables ${VARIABLES} \
            2>&1 | tee -a "${LOG_DIR}/03_analysis_${CAL_NAME}.log"
        
        print_success "Analysis completed for ${CAL_NAME}"
    done
    
    print_success "All analysis completed!"
else
    print_info "Skipping analysis and plotting"
fi

# =============================================================================
# Summary
# =============================================================================
print_header "CALIBRATION WORKFLOW COMPLETED"
echo ""
print_info "Summary:"
print_info "  Timestamp:          ${TIMESTAMP}"
print_info "  Log directory:      ${LOG_DIR}"
print_info "  Model used:         ${MODEL_DIR}"
print_info "  Calibration runs:   ${NUM_CAL_RESULTS}"
print_info "  Calibration period: ${CAL_START} to ${CAL_END}"
print_info "  Validation period:  ${VAL_START} to ${VAL_END}"
print_info "  Full time range:    ${FULL_START} to ${FULL_END}"
echo ""
print_info "Results:"
print_info "  - Calibration results: ${CALIBRATION_OUTPUT}"
print_info "  - Validation results:  ${VALIDATION_RESULTS_DIR}/$(basename ${CALIBRATION_OUTPUT})/"
echo ""
print_info "Generated files per calibration:"
print_info "  - calibrated_parameters.csv"
print_info "  - validation_metrics.csv"
print_info "  - Time series plots"
print_info "  - Scatter plots"
echo ""
print_success "Calibration workflow completed!"
