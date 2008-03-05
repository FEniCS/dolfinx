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
python plot.py "Assembly performance" sequential_poisson_cube.dat parallel_poisson_cube.dat

