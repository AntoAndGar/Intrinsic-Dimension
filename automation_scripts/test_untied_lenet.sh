#!/bin/bash

# test mnist untied lenet model
for dim in {600,700,800,900,1000}
do
    echo dir_"$dim"
    python ./main.py -id $dim -opt 'sgd' -lr 0.01 -arch untied_lenet
done