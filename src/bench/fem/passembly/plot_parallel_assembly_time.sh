#!/bin/sh

if [ ! -r sequential_poisson_cube.dat ] 
then
echo "Run sequential_poisson_cube.sh before running this"
exit
fi
if [ ! -r parallel_poisson_cube.dat ] 
then
echo "Run parallel_poisson_cube.sh before running this"
exit
fi
if [ ! -r parallel_poisson_pdofmap_cube.dat ] 
then
echo "Run parallel_poisson_pdofmap_cube.sh before running this"
exit
fi
if [ ! -r parallel_poissonpdofmap_sparsity_cube.dat ] 
then
echo "Run parallel_poisson_pdofmap_sparsity_cube.sh before running this"
exit
fi
python plot.py "Poisson3D on UnitCube(50, 50, 50)" sequential_poisson_cube.dat parallel_poisson_cube.dat parallel_pdofmap_poisson_cube.dat parallel_pdofmap_sparsity_poisson_cube.dat 

