#!/bin/bash

echo -e "running simulation set $1"

declare -a threadNums=(
    [0]=1
    [1]=2
    [3]=4
    [4]=8
    [5]=12
)

#CPU testing for given array
for i in ${threadNums[@]}
do
python performanceTables.py -b -o $1 --compute cpu -t "$i"
done

python performanceTables.py -b -o $1 --compute cpu
python performanceTables.py -b -o $1 --compute gpu
