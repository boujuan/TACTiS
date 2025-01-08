#!/bin/bash
#SBATCH -p all_gpu.p
#SBATCH -N1
#SBATCH -n1
#SBATCH -c32
#SBATCH --mem=64G
#SBATCH --gres=gpu:H100:1
#SBATCH -t 20:0:0
#SBATCH -o demo_output.txt
#SBATCH -e demo_errors.txt

# Load modules
module load hpc-env/13.1
module load CUDA/12.4.0
module load Mamba/24.3.0-0

eval "$(conda shell.bash hook)"
conda activate tactis_cuda

python gluon_fred_md_forecasting.py

# sbatch run_demo.sh
