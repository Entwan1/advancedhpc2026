#!/bin/bash
rm output/*

echo 'Computation time for a small image:'

python script_CPU.py input/small-cat.jpg output/result_CPU_small-cat.jpg
python script_GPU.py input/small-cat.jpg output/result_GPU_small-cat.jpg

echo 'Computation time for a big image:'

python script_CPU.py input/big-cat.jpg output/result_CPU_big-cat.jpg
python script_GPU.py input/big-cat.jpg output/result_GPU_big-cat.jpg