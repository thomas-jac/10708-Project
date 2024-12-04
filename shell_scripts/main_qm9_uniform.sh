#!/bin/bash
 
#::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#                     Slurm Construction Section
#::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

# job name
#SBATCH --job-name=train_qm9_uniform
# partition (queue) declaration
# Department choices: dept_cpu, dept_gpu, any_cpu, any_gpu, big_memory
# Group choices: bahar_gpu, benos, benos_gpu, camacho_gpu, chakra_24, chakra_gpu
#SBATCH --partition=dept_gpu

# number of requested nodes
#SBATCH --nodes=1

# number of tasks
#SBATCH --ntasks=1

# number of requested cores
#SBATCH --ntasks-per-node=4

# request a GPU
#SBATCH --gres=gpu:1

# request a specific node
#SBATCH --nodelist=g023

# call a Slurm Feature
# #SBATCH --constraint="M20|M24|M40|M46"
# #SBATCH --constraint=4C
# #SBATCH --constraint="TitanV|V100|Turing|gtx1080Ti|Xp"        #"|gtx1080"

# requested runtime
# #SBATCH --time=00:05:00 

# standard output & error
#SBATCH --error=train_qm9_uniform.err
#SBATCH --output=train_qm9_uniform.out

# send email about job start and end
#SBATCH --mail-user=thj19@pitt.edu
#SBATCH --mail-type=ALL

#::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#                     User Construction Section
#::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

# eval "$(conda shell.bash hook)"
# module load anaconda
# mamba init
# export PYTHONUNBUFFERED=TRUE
# source activate cellvit_2
export PYTHONUNBUFFERED=TRUE
# mamba activate pannet_multiplexed
source activate cellvit_2
# python3 /net/dali/home/uttam/thj19/10708-Project/GeoLDM/main_qm9.py --n_epochs 3000 --n_stability_samples 1000 --diffusion_noise_schedule polynomial_2 --diffusion_noise_precision 1e-5 --diffusion_steps 1000 --diffusion_loss_type l2 --batch_size 64 --nf 256 --n_layers 9 --lr 1e-4 --normalize_factors '[1,4,10]' --test_epochs 20 --ema_decay 0.9999 --train_diffusion --trainable_ae --latent_nf 1 --exp_name geoldm_qm9 --wandb_usr tjacob
python3 /net/dali/home/uttam/thj19/10708-Project/GeoLDM/main_qm9.py --n_epochs 500 --n_stability_samples 1000 --diffusion_noise_schedule uniform --diffusion_noise_precision 1e-5 --diffusion_steps 1000 --diffusion_loss_type l2 --batch_size 64 --nf 256 --n_layers 9 --lr 1e-4 --normalize_factors '[1,4,10]' --test_epochs 20 --ema_decay 0.9999 --train_diffusion --trainable_ae --latent_nf 1 --exp_name geoldm_qm9_uniform
