#!/bin/bash
# EEGNet within-subject parallel training launcher
# Usage: bash run_parallel_train.sh
#
# GPU 0, GPU 1에 각각 4개씩 동시 실행 (총 8개 병렬)
# sub-22는 이미 실행 중이므로 제외

set -euo pipefail
cd /local_raid1/03_user/myyu/EEG_decoder

eval "$(conda shell.bash hook)"
conda activate eegpt310
export PYTHONUNBUFFERED=1

LOG_DIR="EEGNet/within_logs/500Hz_t16_s4_w16/train_logs"
mkdir -p "$LOG_DIR"

# All 56 subjects except sub-22 (already running)
ALL_SUBJECTS=(
  sub-14 sub-15 sub-16 sub-17 sub-20 sub-21 sub-23 sub-24
  sub-26 sub-27 sub-29 sub-30 sub-31 sub-32 sub-33 sub-34
  sub-36 sub-37 sub-39 sub-40 sub-41 sub-43 sub-44 sub-45
  sub-46 sub-47 sub-50 sub-51 sub-53 sub-55 sub-56 sub-57
  sub-58 sub-59 sub-60 sub-62 sub-63 sub-64 sub-65 sub-66
  sub-67 sub-68 sub-69 sub-70 sub-71 sub-72 sub-73 sub-74
  sub-75 sub-76 sub-78 sub-79 sub-80 sub-82 sub-83 sub-84
)

N_PER_GPU=4
N_WORKERS=4

# Split subjects across GPUs (alternating)
GPU0_SUBJECTS=()
GPU1_SUBJECTS=()
for i in "${!ALL_SUBJECTS[@]}"; do
    if (( i % 2 == 0 )); then
        GPU0_SUBJECTS+=("${ALL_SUBJECTS[$i]}")
    else
        GPU1_SUBJECTS+=("${ALL_SUBJECTS[$i]}")
    fi
done

echo "========================================"
echo " EEGNet Parallel Training Launcher"
echo "========================================"
echo "[*] Total subjects: ${#ALL_SUBJECTS[@]}"
echo "[*] GPU 0: ${#GPU0_SUBJECTS[@]} subjects (${N_PER_GPU} concurrent)"
echo "[*] GPU 1: ${#GPU1_SUBJECTS[@]} subjects (${N_PER_GPU} concurrent)"
echo "[*] Workers per job: ${N_WORKERS}"
echo "[*] Logs: ${LOG_DIR}/<subject>.log"
echo "========================================"

run_subject() {
    local gpu_id=$1
    local subj=$2
    echo "[START] ${subj} on GPU ${gpu_id} ($(date '+%H:%M:%S'))"
    CUDA_VISIBLE_DEVICES=${gpu_id} python EEGNet/EEGNet_total.py \
        --no_resume --patch_size 16 --subject "${subj}" \
        --devices 0 --num_workers ${N_WORKERS} \
        > "${LOG_DIR}/${subj}.log" 2>&1
    echo "[DONE]  ${subj} on GPU ${gpu_id} ($(date '+%H:%M:%S'))"
}
export -f run_subject
export LOG_DIR N_WORKERS

# GPU 0 jobs (background)
printf '%s\n' "${GPU0_SUBJECTS[@]}" | xargs -P ${N_PER_GPU} -I{} bash -c "run_subject 0 {}" &
PID_GPU0=$!

# GPU 1 jobs (background)
printf '%s\n' "${GPU1_SUBJECTS[@]}" | xargs -P ${N_PER_GPU} -I{} bash -c "run_subject 1 {}" &
PID_GPU1=$!

echo ""
echo "[*] GPU0 master PID: ${PID_GPU0}"
echo "[*] GPU1 master PID: ${PID_GPU1}"
echo "[*] Monitor: tail -f ${LOG_DIR}/*.log"
echo "[*] GPU usage: watch nvidia-smi"
echo ""

wait ${PID_GPU0} ${PID_GPU1}
echo ""
echo "[!] All ${#ALL_SUBJECTS[@]} subjects completed! ($(date))"
