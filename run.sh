#!/bin/bash

nps=(1 2 4 8 16 32)
# nps=(1) # Debug

filename="mm"

delimiter="--------------------------------------------------------------------------------"

mpicc mpi_matrix.c -o $filename

[ ! -f $filename ] || for npi in ${!nps[@]}; do
    if [ $npi -eq 0 ]; then
        echo $delimiter
    fi
    echo "${nps[$npi]} processes:"
    mpirun $filename -np ${nps[$npi]}
    echo $delimiter
done