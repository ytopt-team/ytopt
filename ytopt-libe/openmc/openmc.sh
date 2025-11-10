#!/bin/bash -x

p0="#P0"
pp="openmc"

if [ "$p0" = "$pp" ] 
then
 	openmc  --event -i #P1 -b #P2 -m #P3
else
 	openmc-queueless --event -i #P1 -b #P2 
fi
