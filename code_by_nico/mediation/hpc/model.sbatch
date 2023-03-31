#!/bin/bash

#SBATCH -J bsem2
#SBATCH -p general
#SBATCH -o bsem2_%j.txt
#SBATCH -e bsem2error_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=nescoba@iu.edu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=10:00:00
#SBATCH --mem=128G
#SBATCH -A general
#SBATCH --cpus-per-task=4

load module R

#Run your program
srun Rscript --vanilla bsem2.R