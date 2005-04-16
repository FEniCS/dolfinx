#!/bin/sh

# Compute reference solutions on different meshes
# k = 1/n, 1/2n, 1/4n, ...

nlist="1 2 4 8 16"
num_k="9"

MFILE="tmp.m"

rm -r $MFILE

for n in $nlist; do

    echo ""
    echo "Computing reference solution for n = $n"
    echo "-----------------------------------------"

    k=`echo $n | awk '{ print 2.0/$n }'`
    filename="solution_$n.data"
    rm -f $filename
    for i in `seq $num_k`; do
	k=`echo $k | awk '{ print $k/2.0 }'`
	echo "Time step k = $k"

	./dolfin-bench-ode cg 3 $n $k
    	cat solution.data >> $filename
    done

    echo "load $filename" >> $MFILE
    echo "max(abs(diff(solution_$n, 2))')" >> $MFILE

done

octave $MFILE
