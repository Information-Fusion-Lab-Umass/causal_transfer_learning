#!/bin/bash
#BATCH --job-name=test
#SBATCH --output=res_%j.txt  # output file
#SBATCH -e res_%j.err        # File to which STDERR will be written
#SBATCH --partition 1080ti-long # Partition to submit to
#SBATCH --ntasks=1
#SBATCH --time=10:00         # Runtime in D-HH:MM
#SBATCH --mem-per-cpu=20240    # Memory in MB per cpu allocated
#SBATCH --gres gpu:4
#. venv/bin/activate
PYTHONPATH=$PWD python codes/test_gypsum.py
