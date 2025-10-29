# PARCIAL2-PARALELA

Ejecutar Cleanstream:

mpirun -np 8 python3 clean_mpi.py dirty_data.csv


gcc -O2 generate_dirty_data.c -o generate_dirty_data
mpicc -fopenmp cleanstream.c -O2 -o cleanstream -lm

Generar filas, ./generate_dirty_data 200000 dirty_200k.csv 


Ejecutar secuencial (baseline, rank 0 hará la pasada secuencial y además la paralela con 1 proceso):

mpirun -np 1 ./cleanstream dirty_200k.csv 

mpirun -np 4 ./cleanstream dirty_200k.csv

