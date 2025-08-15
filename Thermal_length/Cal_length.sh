#!/bin/bash
for j in {1..20..2}
do
	for k in {1..20..2}
	do
		echo $j $k
		python3 Cal_thermal_length.py 700 $j $k
	done
done
