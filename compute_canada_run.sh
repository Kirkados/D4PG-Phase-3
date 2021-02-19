#!/bin/bash
#SBATCH --cpus-per-task=48   # number of CPUs requested. Beluga: 40; Niagara: 40 (80 for hyperthreading); Cedar: 32 (or 48); Graham: 32
#SBATCH --time=4-00:00:00   # time (DD-HH:MM:SS) (Max on Beluga/Cedar/Graham: 7 days; Niagara 1 day)
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

##Niagara commands if fresh
   #module load NiaEnv/2019b
   #module load python/3.7.9
   #module load gcc/8.3.0
   #module load geos
   #virtualenv --system-site-packages $HOME/env   
   # source $HOME/env/bin/activate   
   #pip3 install --upgrade pip
   #pip3 install -r requirements.txt
   #pip3 install PyVirtualDisplay
   
## Other system commands if fresh
   #module load StdEnv/2020 # if you haven't upgraded to 2020 as the default yet
module load python/3.7.7
module load scipy-stack
module load geos
virtualenv --no-download $SCRATCH/env 
source $SCRATCH/env/bin/activate   
pip3 install --no-index --upgrade pip
pip3 install --no-index -r requirements.txt
pip3 install EasyProcess-0.3-py2.py3-none-any.whl
pip3 install PyVirtualDisplay-2.0-py2.py3-none-any.whl

source $SCRATCH/env/bin/activate


# Submit this job from the D4PG Folder
tensorboard --logdir=$SCRATCH/D4PG-Phase-3/Tensorboard/Current/ --host=0.0.0.0 &
python3 -u main.py