#!/bin/bash -l
#SBATCH --job-name="penny_multi_qubits"
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-task=1
#SBATCH --output=%x%j.out
#SBATCH --error=%x%j.err
#SBATCH -p gpu
#SBATCH -q default
#SBATCH --time=48:00:00
#SBATCH -A lxp

#module load env/staging/2023.1
module load PennyLane-Lightning-GPU/0.37.0-foss-2023a-CUDA-12.2.0
#module load TensorFlow/2.15.1-foss-2023a-CUDA-12.1.1
module load matplotlib/3.7.2-gfbf-2023a  Seaborn/0.13.2-gfbf-2023a

#export NCCL_SOCKET_IFNAME=ib0
export UCX_MAX_RNDV_RAILS=1
export NCCL_CROSS_NIC=1

#srun --ntasks-per-node=4 --gpus-per-task=1 python3 quanvolution_mpi.py

#output_file="gpu_timing_data.txt"

# Clear the output file if it exists
#> $output_file

srun --ntasks-per-node=4 --gpus-per-task=1 python3 multi_qubits2.py
