#!/bin/bash
for i in {0..403}
do
       echo $i	
	./ODE_solver $i
done
