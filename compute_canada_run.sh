#!/bin/bash
#SBATCH --cpus-per-task=80   # number of CPUs requested. Beluga: 40; Niagara: 40 (80 for hyperthreading); Cedar: 32 (or 48); Graham: 32
#SBATCH --time=0-01:00:00   # time (DD-HH:MM:SS) (Max on Beluga/Cedar/Graham: 7 days; Niagara 1 day)
#SBATCH --nodes=1
##SBATCH --mem-per-cpu=4G
#SBATCH --mem=0 # take all memory
#SBATCH --account=def-sulrich
#SBATCH --output=%j.out
#SBATCH --mail-user=khovell@gmail.com
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-type=REQUEUE
#SBATCH --mail-type=ALL
NIAGARA=true

if [NIAGARA] # set to True if using Niagara
then
   module load CCEnv # uncomment if on Niagara, comment otherwise
   module load StdEnv/2020 # uncomment if on Niagara, comment otherwise
fi
module load python/3.7.7
module load scipy-stack
module load geos
if [NIAGARA]
then
   virtualenv --no-download virtualEnv
   source virtualEnv/bin/activate
else
   virtualenv --no-download $SLURM_TMPDIR/env   
   source $SLURM_TMPDIR/env/bin/activate
fi
pip3 install --no-index --upgrade pip
pip3 install --no-index -r requirements.txt
pip3 install EasyProcess-0.3-py2.py3-none-any.whl
pip3 install PyVirtualDisplay-2.0-py2.py3-none-any.whl

tensorboard --logdir=$SCRATCH/D4PG-Phase-3/Tensorboard/Current/ --host=0.0.0.0 &
python3 -u main.py