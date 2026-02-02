#!/bin/bash
/usr/local/slurm/bin/salloc -N 1 --gres=gpu:v100:2 -J interactive -p gpuserver -t 8:00:00 /usr/local/slurm/bin/srun --pty /bin/bash -l