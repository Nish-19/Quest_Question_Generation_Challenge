#!/bin/bash
echo "Incremental values for the loss weight lambda"

const=0.5

for i in $(seq .1 .2 1); 
do 
    if (( $(echo "$i != $const" |bc -l) ));
    then
        echo 'Training with Lambda:' $i
        python -m code.finetune.finetune_data_aug -W -E 10 -D 3 -MT T -MN google/flan-t5-large -N flan_t5_large -LAM $i 
    fi
done