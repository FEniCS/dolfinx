#!/bin/sh

alpha='5'
betalist='1 2 3 4'
mlist='1 2 3'

for beta in $betalist; do

    echo "beta = $beta"

    for m in $mlist; do
	
	echo "  m = $m"
	prefix="ces-$m-$m-$alpha-$beta"
	./dolfin-ode-homotopy-ces $m $m $alpha $beta > $prefix.log
	mv solution.data $prefix.data

    done

done
