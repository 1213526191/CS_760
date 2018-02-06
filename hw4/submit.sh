#!/bin/bash
#SBATCH --mail-type=ALL
#SBATCH --mail-user=yli768@wisc.edu
#SBATCH -p long

export PATH=/workspace/software/bin:$PATH
export R_LIBS="/s/R"
export TMPDIR=/data1/stat479
module load course/stat479

python CV.py