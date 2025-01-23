#!/bin/bash
#SBATCH -p cfdg.p
#SBATCH -N1
#SBATCH -n1
#SBATCH -c32
#SBATCH --mem=64G
#SBATCH --gres=gpu:H100:1
#SBATCH -t 100:0:0
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
# sbatch -p all_gpu.p -N1 -n1 -c32 --mem=64G --gres=gpu:H100:1 -t 20:0:0 run_model.sh