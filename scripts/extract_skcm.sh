#!/bin/sh
#
#SBATCH --job-name="extract_skcm"
#SBATCH --partition=compute
#SBATCH --time=4:00:00
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=1G
#SBATCH --cores=4

module load 2023r1
module load py-pip

cd /home/wsung/SignatureCharacterization

python src/extract_skcm.py