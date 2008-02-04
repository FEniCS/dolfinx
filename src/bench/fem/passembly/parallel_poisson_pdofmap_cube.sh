#!/bin/bash
outfile=parallel_poisson_pdofmap_cube.dat
num_procs=4
iter=1
num_cells=30

echo "Parallel Poisson with parallel DofMap" > $outfile
echo "Processes Time" >> $outfile
for ((  i=1 ;  i<=$num_procs;  i++  ))
do
  echo "Running:
  mpirun -n $i ./dolfin-pmesh-test --cells $num_cells --num_iterations $iter --resultfile $outfile
  "
  mpirun -n $i ./dolfin-pmesh-test --cells $num_cells --num_iterations $iter --resultfile $outfile
done
