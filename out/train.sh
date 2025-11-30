#!/bin/bash

# Training script for vequinox

NUM_EPOCHS=${1:-5}
NUM_SAMPLES=${2:-500}
OUTPUT_DIR=${3:-"./vequinox-checkpoints"}

# Create logs directory if it doesn't exist
mkdir -p logs

# Generate timestamp for log file
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="logs/train_${TIMESTAMP}.log"

echo "Starting training..."
echo "  Epochs: $NUM_EPOCHS"
echo "  Samples: $NUM_SAMPLES"
echo "  Output dir: $OUTPUT_DIR"
echo "  Log file: $LOG_FILE"

python out/main.py \
    --num_train_epochs "$NUM_EPOCHS" \
    --num_samples "$NUM_SAMPLES" \
    --output_dir "$OUTPUT_DIR" \
    2>&1 | tee "$LOG_FILE"
