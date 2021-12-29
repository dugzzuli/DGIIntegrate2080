‘’‘bash run.sh’‘
#!/bin/bash
sc=10.0

for lr in 0.01 0.001 0.0001 0.00001
do
    for l2_coef in 0.01 0.001 0.0001 0.00001
    do
        for reg_coef in 0.01 0.001 0.0001 0.00001
        do
            for hid_units in 256 512
            do
                for i in 1
                do
                    echo "--lr $lr --l2_coef $l2_coef --reg_coef $reg_coef --hid_units $hid_units --sc $sc"
                    python runMainDMGI.py --lr $lr --l2_coef $l2_coef --reg_coef $reg_coef --hid_units $hid_units --sc $sc
                done
            done
        done
    done
done