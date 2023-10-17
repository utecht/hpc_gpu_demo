#!/bin/bash
#SBATCH --partition=gpu
source activate ~/.bashrc
conda activate tensorflow_test
python hello_ml.py
