#!/bin/sh
#
#SBATCH --job-name="extract_stad"
#SBATCH --partition=compute
#SBATCH --time=10:00:00
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=6
#SBATCH --mem-per-cpu=2G
#SBATCH --cores=4

module load 2023r1
module load py-pip

cd /home/wsung/SignatureCharacterization

python src/extract_stad.py