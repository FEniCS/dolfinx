#!/bin/sh

# Compute reference solutions on different meshes
# k = 1/n, 1/2n, 1/4n, ...

nlist="1 2 4 8"
num_k="3"

for n in $nlist; do

    echo ""
    echo "Computing reference solution for n = $n"
    echo "-----------------------------------------"

    k=`echo $n | awk '{ print 2.0/$n }'`
    for i in `seq $num_k`; do
	k=`echo $k | awk '{ print $k/2.0 }'`
	echo "Time step k = $k"

	./dolfin-bench-ode cg 1 $n $k
    	mv solution.data solution\_$n\_$i.data
    done

done
