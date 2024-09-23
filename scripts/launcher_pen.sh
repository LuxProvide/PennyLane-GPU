#!/bin/bash -l
#SBATCH --job-name="penny_multi_gpus"
#SBATCH --nodes=64
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

export NCCL_SOCKET_IFNAME=ib0
export UCX_MAX_RNDV_RAILS=1
export NCCL_CROSS_NIC=1

#srun --ntasks-per-node=4 --gpus-per-task=1 python3 quanvolution_mpi.py

#output_file="gpu_timing_data.txt"

# Clear the output file if it exists
#> $output_file

# Define the list of GPU counts
node_counts=(4 8 16 32 64)

for nodes in "${node_counts[@]}"; do
    echo "Running with $nodes nodes..."
    srun -N $nodes --ntasks-per-node=4 --gpus-per-task=1 python3 multi_gpu2.py
done

#echo "Data collection complete. Results saved in $output_file"

#srun -N 1 --ntasks-per-node=1 --gpus-per-task=1 python3plot_gpus.py
#srun --ntasks-per-node=4 --gpus-per-task=1 python3 multi_gpu.py
