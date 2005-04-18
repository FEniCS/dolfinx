#!/bin/sh

# Compute reference solutions on different meshes
# k = 1/2n, 1/4n, 1/8n, ...

nlist="1 2 4 8 16 32 64 128 256 512 1024"
num_k="7"

MFILE="tmp.m"
LOGFILE="reference.log"

rm -f $MFILE
rm -f $LOGFILE

for n in $nlist; do

    echo ""
    echo "Computing reference solution for n = $n"
    echo "-----------------------------------------"

    k=`echo $n | awk '{ print 1.0/$n }'`
    filename="solution_$n.data"
    rm -f $filename
    for i in `seq $num_k`; do
	k=`echo $k | awk '{ print $k/2.0 }'`
	echo "Time step k = $k"

	./dolfin-bench-ode cg 3 $n $k >> $LOGFILE
    	cat solution.data >> $filename
    done

    echo "load $filename" >> $MFILE
    echo "max(abs(diff(solution_$n))')" >> $MFILE

done

octave $MFILE
