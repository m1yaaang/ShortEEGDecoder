#!/bin/bash
# Training 완료 대기 후 자동 inference 실행
# Usage: bash run_after_train.sh

set -euo pipefail
cd /local_raid1/03_user/myyu/EEG_decoder

eval "$(conda shell.bash hook)"
conda activate eegpt310
export PYTHONUNBUFFERED=1

TRAIN_PID_FILE="/tmp/eegnet_train_pids"
LOG_ROOT="EEGNet/within_logs/500Hz_t16_s4_w16"

echo "========================================"
echo " Training → Inference Auto Pipeline"
echo "========================================"

# -------------------------------------------------------
# Wait for training to finish
# -------------------------------------------------------
echo "[*] Waiting for parallel training to complete..."
echo "[*] Monitoring: ${LOG_ROOT}/train_logs/"

while true; do
    # Check if any EEGNet_total.py training processes are still running
    n_running=$(pgrep -f "python.*EEGNet_total.py.*--no_resume" 2>/dev/null | grep -v pgrep | wc -l)
    if [[ $n_running -eq 0 ]]; then
        echo "[*] No training processes detected. Training complete!"
        break
    fi

    # Progress report
    n_subjects=$(find "${LOG_ROOT}" -maxdepth 1 -type d -name "sub-*" | wc -l)
    echo "[$(date '+%H:%M:%S')] Training in progress: ${n_running} processes, ${n_subjects} subjects with data"
    sleep 60
done

echo ""
echo "========================================"
echo " Starting Parallel Inference"
echo "========================================"

# -------------------------------------------------------
# Run inference for all subjects
# -------------------------------------------------------
bash run_inference_parallel.sh

echo ""
echo "========================================"
echo " Pipeline Complete! ($(date))"
echo "========================================"
