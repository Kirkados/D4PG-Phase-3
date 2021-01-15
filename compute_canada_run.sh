#!/bin/bash
#SBATCH --ntasks=4          # number of processes
#SBATCH --mem-per-cpu=4G    # memory
#SBATCH --account=def-sulrich
#SBATCH --time=0-00:01:00   # time (DD-HH:MM:SS)
#SBATCH --output=%N-%j.out  # %N for node name, %j for jobID

module load python/3.7.4
virtualenv --no-download $SLURM_TEMPDIR/env
source $SLURM_TEMPDIR/env/bin/activate
pip3 install --no-index --upgrade pip

pip3 install --no-index -f requirements.txt
pip3 install PyVirtualDisplay-2.0-py2.py3-none-any.whl
pip3 install EasyProcess-0.3-py2.py3-none-any.whl

python3 main.py