#!/bin/bash
# EEGNet Within-Subject Inference - Parallel Launcher
#
# Usage:
#   # sub-22만 테스트
#   bash run_inference_parallel.sh --subject sub-22
#
#   # 전체 subject 병렬 inference
#   bash run_inference_parallel.sh
#
#   # 특정 subject들
#   bash run_inference_parallel.sh --subject sub-22 sub-23 sub-27
#
#   # patch_size 변경
#   bash run_inference_parallel.sh --patch_size 64
#
#   # 기존 결과 무시하고 재실행
#   bash run_inference_parallel.sh --force
#
#   # cross-subject TGM만 재생성
#   bash run_inference_parallel.sh --cross-subject-only

set -euo pipefail
cd /local_raid1/03_user/myyu/EEG_decoder

eval "$(conda shell.bash hook)"
conda activate eegpt310
export PYTHONUNBUFFERED=1

# -------------------------------------------------------
# Default parameters
# -------------------------------------------------------
PATCH_SIZE=16
STRIDE=4
TIME_BIN=16
N_PER_GPU=2          # GPU당 동시 inference 수
NUM_WORKERS=4        # 프로세스당 dataloader workers
BATCH_SIZE=4096
FORCE=""
CROSS_ONLY=""
SUBJECTS=()

# -------------------------------------------------------
# Parse arguments
# -------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case $1 in
        --patch_size) PATCH_SIZE="$2"; shift 2 ;;
        --stride) STRIDE="$2"; shift 2 ;;
        --time_bin) TIME_BIN="$2"; shift 2 ;;
        --n_per_gpu) N_PER_GPU="$2"; shift 2 ;;
        --num_workers) NUM_WORKERS="$2"; shift 2 ;;
        --batch_size) BATCH_SIZE="$2"; shift 2 ;;
        --force) FORCE="--force"; shift ;;
        --cross-subject-only) CROSS_ONLY="--cross-subject-only"; shift ;;
        --subject) shift; while [[ $# -gt 0 && ! "$1" =~ ^-- ]]; do
            SUBJECTS+=("$1"); shift; done ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

LOG_ROOT="EEGNet/within_logs/500Hz_t${TIME_BIN}_s${STRIDE}_w${PATCH_SIZE}"
LOG_DIR="${LOG_ROOT}/inference_logs"
mkdir -p "$LOG_DIR"

# -------------------------------------------------------
# Cross-subject only mode
# -------------------------------------------------------
if [[ -n "$CROSS_ONLY" ]]; then
    echo "[*] Cross-subject TGM only mode"
    python EEGNet/EEGNet_inference_total.py \
        --cross-subject-only \
        --patch_size "$PATCH_SIZE" --stride "$STRIDE" --time_bin "$TIME_BIN"
    exit 0
fi

# -------------------------------------------------------
# Discover subjects from trained checkpoints
# -------------------------------------------------------
if [[ ${#SUBJECTS[@]} -eq 0 ]]; then
    # Auto-discover all trained subjects
    for d in "${LOG_ROOT}"/sub-*/; do
        [[ -d "$d" ]] || continue
        subj=$(basename "$d")
        SUBJECTS+=("$subj")
    done
fi

if [[ ${#SUBJECTS[@]} -eq 0 ]]; then
    echo "[!] No subjects found in ${LOG_ROOT}"
    exit 1
fi

# Sort subjects
IFS=$'\n' SUBJECTS=($(sort <<<"${SUBJECTS[*]}")); unset IFS

echo "========================================"
echo " EEGNet Parallel Inference Launcher"
echo "========================================"
echo "[*] patch_size=${PATCH_SIZE}, stride=${STRIDE}, time_bin=${TIME_BIN}"
echo "[*] Subjects: ${#SUBJECTS[@]}"
echo "[*] GPU 0,1: ${N_PER_GPU} concurrent each"
echo "[*] Workers/process: ${NUM_WORKERS}"
echo "[*] Logs: ${LOG_DIR}/<subject>.log"
echo "========================================"

# -------------------------------------------------------
# Run inference per subject (called by xargs)
# -------------------------------------------------------
run_subject() {
    local gpu_id=$1
    local subj=$2
    echo "[START] ${subj} on GPU ${gpu_id} ($(date '+%H:%M:%S'))"
    CUDA_VISIBLE_DEVICES=${gpu_id} python EEGNet/EEGNet_inference_total.py \
        --subject "${subj}" \
        --patch_size "${PATCH_SIZE}" --stride "${STRIDE}" --time_bin "${TIME_BIN}" \
        --batch_size "${BATCH_SIZE}" --num_workers "${NUM_WORKERS}" \
        ${FORCE} \
        > "${LOG_DIR}/${subj}.log" 2>&1
    echo "[DONE]  ${subj} on GPU ${gpu_id} ($(date '+%H:%M:%S'))"
}
export -f run_subject
export LOG_DIR PATCH_SIZE STRIDE TIME_BIN BATCH_SIZE NUM_WORKERS FORCE

# -------------------------------------------------------
# Split subjects across GPUs and run
# -------------------------------------------------------
GPU0_SUBJECTS=()
GPU1_SUBJECTS=()
for i in "${!SUBJECTS[@]}"; do
    if (( i % 2 == 0 )); then
        GPU0_SUBJECTS+=("${SUBJECTS[$i]}")
    else
        GPU1_SUBJECTS+=("${SUBJECTS[$i]}")
    fi
done

# GPU 0
if [[ ${#GPU0_SUBJECTS[@]} -gt 0 ]]; then
    printf '%s\n' "${GPU0_SUBJECTS[@]}" | \
        xargs -P "${N_PER_GPU}" -I{} bash -c "run_subject 0 {}" &
    PID_GPU0=$!
fi

# GPU 1
if [[ ${#GPU1_SUBJECTS[@]} -gt 0 ]]; then
    printf '%s\n' "${GPU1_SUBJECTS[@]}" | \
        xargs -P "${N_PER_GPU}" -I{} bash -c "run_subject 1 {}" &
    PID_GPU1=$!
fi

# Wait for all inference to finish
wait

echo ""
echo "[*] All per-subject inference done. Generating cross-subject analysis..."

# -------------------------------------------------------
# Cross-subject TGM (single process, after all subjects done)
# -------------------------------------------------------
python EEGNet/EEGNet_inference_total.py \
    --cross-subject-only \
    --patch_size "$PATCH_SIZE" --stride "$STRIDE" --time_bin "$TIME_BIN"

echo ""
echo "[!] All done! ($(date))"
echo "[*] Results: ${LOG_ROOT}/cross_subject_analysis/"
