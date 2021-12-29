#!/bin/bash
sc=10.0
for lambdapra in 10 1 0.1 0.01
do
  for lr in 0.01 0.001
  do
    for l2_coef in 0001
    do
        for reg_coef in 0001
        do
            for T in 1 3 5 10 20 50
            do
                for i in 1
                do
                    echo "--lr $lr --l2_coef $l2_coef --reg_coef $reg_coef --hid_units $hid_units --sc $sc"
                    python runMainDMGIDEC.py --lr $lr --l2_coef $l2_coef --reg_coef $reg_coef --T $T --sc $sc
                done
            done
        done
    done
  done

done
