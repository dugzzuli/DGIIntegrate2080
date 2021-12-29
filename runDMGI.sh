#!/bin/bash
sc=10.0

for lr in 0.01 0.001 0.0001
do
    for cls_reg in 0.01 0.001 0.0001 0.00001
    do
        for l2_coef in 0.01 0.001 0.0001 0.00001
        do
            for hid_units in 128 256 512
            do
            
                python runMainDMGIN2.py --lr $lr --cls_reg $cls_reg --l2_coef $l2_coef --hid_units $hid_units
            
            done
        done
    done
done

