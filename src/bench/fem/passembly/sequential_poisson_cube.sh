#!/bin/bash
outfile=sequential_poisson_cube.dat
num_procs=4
iter=1
num_cells=30

echo "Sequential Poisson" > $outfile
echo "Processes Time" >> $outfile
for ((  i=1 ;  i<=$num_procs;  i++  ))
do
  echo "Running:
  mpirun -n $i ./dolfin-pmesh-test --sequential --cells $num_cells --num_iterations $iter --resultfile $outfile
  "
  mpirun -n $i ./dolfin-pmesh-test --sequential --cells $num_cells --num_iterations $iter --resultfile $outfile
done
