#!/bin/bash

#SBATCH -A MST111038
#SBATCH --job-name=tts_train       # Job name
#SBATCH --output=/work/smartllm172/log/tts_train_%j.out # Output file name (%j expands to jobID)
#SBATCH --error=/work/smartllm172/log/tts_train_%j.err  # Error file name (%j expands to jobID)
#SBATCH --ntasks=1                  # Run a single task
#SBATCH --cpus-per-task=4           # Number of CPU cores per task (increased for data loading)
#SBATCH --mem=32G                   # Memory requirement (increased for large dataset)
#SBATCH --time=20:00:00             # Time limit (HH:MM:SS) - increased for large dataset
#SBATCH --partition=normal            # Partition/queue name
#SBATCH --gpus-per-node=1           # GPU Count
#SBATCH --mail-type=BEGIN,END,FAIL  # Mail events
#SBATCH --mail-user=chenjoachim63@outlook.com

# Print some information about the job
echo "Running on host: $(hostname)"
echo "Starting time: $(date)"
echo "Working directory: $(pwd)"
echo "CUDA devices: $CUDA_VISIBLE_DEVICES"

# Set Conda paths
CONDA_PREFIX="/opt/ohpc/pub/utils/miniconda3/24.11.1"  # Conda installation path
CONDA_DEFAULT_ENV="miniomni"                       # Your Conda environment name

echo "Using Conda from: $CONDA_PREFIX"
echo "Using Conda environment: $CONDA_DEFAULT_ENV"

# Display NVIDIA GPU information
nvidia-smi

# Set up Wandb login
echo -e "\n====== Setting up Wandb ======\n"
export WANDB_API_KEY=$(cat $HOME/.wandb_key)  # Replace with your actual API key
$CONDA_PREFIX/bin/conda run -n "$CONDA_DEFAULT_ENV" python -c "import wandb; wandb.login()"
# Create output directory if it doesn't exist
mkdir -p /work/smartllm172/miniomniTTS/checkpoints

# Run the TTS training script
echo -e "\n====== Running TTS Training ======\n"
$CONDA_PREFIX/bin/conda run -n "$CONDA_DEFAULT_ENV" python /work/smartllm172/miniomniTTS/tts_train.py \
    --model_name "meta-llama/Llama-3.2-1B-Instruct" \
    --dataset_name "/work/smartllm172/GLM-4_dialogue" \
    --num_samples 500000 \
    --num_epochs 1 \
    --batch_size 8 \
    --gradient_accumulation_steps 4 \
    --lr 1e-4 \
    --checkpoint_dir "/work/smartllm172/miniomniTTS/checkpoints" \
    --do_eval

# End of job info
echo -e "\nJob finished: $(date)"
