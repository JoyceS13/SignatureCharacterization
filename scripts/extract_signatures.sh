#!/bin/sh
#
#SBATCH --job-name="extract_signatures"
#SBATCH --partition=compute
#SBATCH --time=14:00:00
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=2G

module load 2023r1
module load py-pip

cd /home/wsung/SignatureCharacterization

python src/extract_all.py