#!/bin/bash
  
let k=${OMPI_COMM_WORLD_RANK}%6
nvidia-smi dmon  -i $k -s p -d 1 -f power${OMPI_COMM_WORLD_RANK}.txt &
dmpid=$!
$1 -m event 
kill $dmpid
sleep 1
./pavg.pl power0.txt power1.txt power2.txt power3.txt power4.txt power5.txt
