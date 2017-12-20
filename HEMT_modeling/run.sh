#!/bin/bash
folder_name="diff_alpha_"
alphas="1 0.5 0.2 0.1 0.05 0.02 0.01 0.005 0.002 0.001"
for i in `seq 1 5`;
do
	mkdir "$folder_name$i"
	let idx=0
	for j in $alphas
	do
		let idx+=1
        python hemt_trainer_example.py "$folder_name$i/HEMT_DC_$idx" 5e4 $j &
    done
    wait
done 