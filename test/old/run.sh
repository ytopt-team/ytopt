#!/bin/sh

python gen_scripts.py

for i in scripts/*.sh
do
echo $i
sh $i
done

