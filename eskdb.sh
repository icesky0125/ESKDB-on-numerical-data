#!/bin/bash
#SBATCH --job-name=ESKDB_HDP
#SBATCH --account=nc23
#SBATCH --time=150:15:00
#SBATCH --mem=15000
#SBATCH --array=1
#SBATCH --output=%A_%a.txt
cd dataContinuous12
arrayfile=`ls | awk -v line=$SLURM_ARRAY_TASK_ID '{if (NR == line) print $0}'`
cd ..
java -Xmx20000m -classpath ./bin:./lib/weka.jar:./lib/commons-math3-3.6.1.jar:./lib/MLTools.jar MDL_R.TwoFoldCV -t $arrayfile -X 5 -K 5 -S SKDB_R -I 1000 -L 2 -E 20 >

