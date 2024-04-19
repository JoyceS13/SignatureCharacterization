#!/bin/sh
#
#SBATCH --job-name="extract_skcm"
#SBATCH --partition=compute
#SBATCH --time=8:00:00
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=6
#SBATCH --mem-per-cpu=4G

module load 2023r1
module load py-pip

cd /home/wsung/SignatureCharacterization

python src/extract_skcm.py