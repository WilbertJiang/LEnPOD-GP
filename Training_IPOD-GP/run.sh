#!/bin/bash
cd ./fem_cpp/build
mpirun -n 20 ./GPU_Core_Train
cd ../..
mpirun -n 20 python3 ComputingAM_flp7.py
python3 save_podmode.py
mpirun -n 20 python3 Compute_P_mode_integration.py
