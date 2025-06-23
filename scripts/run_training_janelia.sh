#!/bin/bash

# Job allocation parameters
NUM_CORES=4
GPU_NUM=1
QUEUE="gpu_h200"
JOB_NAME="autoencoder_train"

# Create a unique run directory with timestamp
RUN_DIR="runs/run_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RUN_DIR"
mkdir -p "$RUN_DIR/checkpoints"
mkdir -p "$RUN_DIR/logs"

# Create a job script
cat > "job_script.sh" << EOF
#!/bin/bash
#BSUB -n $NUM_CORES
#BSUB -gpu "num=$GPU_NUM"
#BSUB -q $QUEUE
#BSUB -J $JOB_NAME
#BSUB -o "$RUN_DIR/logs/%J.out"
#BSUB -e "$RUN_DIR/logs/%J.err"

# Get the compute node hostname
COMPUTE_NODE=\$(hostname)
echo "Running on compute node: \$COMPUTE_NODE" > "$RUN_DIR/logs/node_info.txt"

# Start TensorBoard in the background
echo "Starting TensorBoard..."
tensorboard --bind_all --port 6006 --logdir "$RUN_DIR" > "$RUN_DIR/logs/tensorboard.log" 2>&1 &
TENSORBOARD_PID=\$!
echo "TensorBoard started with PID: \$TENSORBOARD_PID" >> "$RUN_DIR/logs/node_info.txt"

# Print instructions for accessing TensorBoard
echo "To access TensorBoard:" >> "$RUN_DIR/logs/node_info.txt"
echo "1. From your local machine, run: ssh -L 6006:\$COMPUTE_NODE:6006 your_username@login_node" >> "$RUN_DIR/logs/node_info.txt"
echo "2. Then open http://localhost:6006 in your browser (local machine)" >> "$RUN_DIR/logs/node_info.txt"

# Run the training script
echo "Starting training..."
apptainer exec --nv \\
    --bind /groups/betzig/betziglab/CellObservatoryData/:/CellObservatoryData \\
    --bind \$PWD:/workspace \\
    ../develop_torch_cuda_12_8.sif \\
    python train.py \\
    --num_workers 4 \\
    --log_dir "$RUN_DIR" \\
    --checkpoint_dir "$RUN_DIR/checkpoints"

# Cleanup: kill TensorBoard when training is done
kill \$TENSORBOARD_PID
EOF

# Make the job script executable
chmod +x "job_script.sh"

# Submit the job
JOB_ID=$(bsub < job_script.sh | grep -oP '(?<=Job <)\d+(?=>)')

# Print job information
echo "Job submitted with ID: $JOB_ID"
echo "Run directory: $RUN_DIR"
echo "To monitor the job:"
echo "  bjobs $JOB_ID"
echo "  bpeek $JOB_ID"
echo "To view logs:"
echo "  cat $RUN_DIR/logs/%J.out"
echo "  cat $RUN_DIR/logs/%J.err"
echo "  cat $RUN_DIR/logs/tensorboard.log"
echo "  cat $RUN_DIR/logs/node_info.txt"
echo "To kill the job:"
echo "  bkill $JOB_ID" 