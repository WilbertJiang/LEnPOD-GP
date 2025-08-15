#!/bin/bash
for i in {550..800..50}
do
	for j in {1..20..2}
	do
		for k in {1..20..2}
		do
			echo $i $j $k
			mpirun -n 30 python3 FEM_steady_state.py $i $j $k
		done
	done
done
