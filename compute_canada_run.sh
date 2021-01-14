#!/bin/bash
#SBATCH --ntasks=4          # number of processes
#SBATCH --mem-per-cpu=4G    # memory
#SBATCH --account=def-sulrich
#SBATCH --time=0-00:01:00   # time (DD-HH:MM:SS)
#SBATCH --output=%N-%j.out  # %N for node name, %j for jobID
python3 main.py