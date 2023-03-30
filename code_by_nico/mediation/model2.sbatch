#!/bin/bash

#SBATCH -J bsem2
#SBATCH -p general
#SBATCH -o bsem2_%j.txt
#SBATCH -e bsem2error_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=nescoba@iu.edu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=02:00:00
#SBATCH --mem=2G
#SBATCH -A general
#SBATCH --cpus-per-task=4

#Run your program
R bsem2.R