#!/bin/sh
#
# Copyright (C) 2004 Johan Jansson.
# Licensed under the GNU GPL Version 2.
#
# Modified by Anders Logg, 2004.
#
# Note: /usr/bin/time (Debian package time) is needed
# to avoid using the bash built-in time, which is
# difficult to redirect.

function runsingle()
{
    /usr/bin/time -f"$TIMEFORMAT" -otlog ./dolfin-ode-perf-test $1 $2 $3 $4 1> log1 2> log2
    TIME=$( cat tlog )
    echo "Elapsed time: "$TIME
    echo -n $( echo $TIME | awk '{ print $4 }' ) " + " $( echo $TIME | awk '{ print $6 }' ) >> timings.m
    echo -n " " >> timings.m
    echo
}

function run()
{
    #n="10 20 40 100 200 400 1000 2000 4000 10000 20000 50000 100000"
    n="10 20 40 100 200 400 1000"
    
    if [ "$1" = "" ]; then
	M=100
    else
	M=$1
    fi
    
    if [ "$2" = "" ]; then
	b=100
    else
	b=$2
    fi

    echo "---------------------------------------------------------------------------"    
    echo "M = "$M
    echo "b = "$b
    echo

    echo "n = [ " $n "];" > timings.m
    echo -n "t1 = [" >> timings.m

    # Run multi-adaptive test
    echo "Running multi-adaptive performance test"
    echo "---------------------------------------"
    echo

    for i in $n; do
	echo "mdG(0) n = " $i
	runsingle $i $M $b mdg	
    done

    echo "];" >> timings.m
    echo "k1 = " $( echo "timings; lsquares(n, t1)(1)" | octave -q | awk '{ print $3 }' ) ";" >> timings.m
    echo "m1 = " $( echo "timings; lsquares(n, t1)(2)" | octave -q | awk '{ print $3 }' ) ";" >> timings.m
    echo
    echo -n "t2 = [" >> timings.m

    # Run mono-adaptive test
    echo "Running mono-adaptive performance test"
    echo "--------------------------------------"
    echo

    for i in $n; do
	echo "dG(0) n = " $i
	runsingle $i $M mono
    done

    echo "];" >> timings.m
    echo "k2 = " $( echo "timings; lsquares(n, t2)(1)" | octave -q | awk '{ print $3 }' ) ";" >> timings.m
    echo "m2 = " $( echo "timings; lsquares(n, t2)(2)" | octave -q | awk '{ print $3 }' ) ";" >> timings.m

}

VERSION=`../../../../config/dolfin-config --version`
DATE=`date`
CFLAGS=`../../../../config/dolfin-config --cflags`
TIMEFORMAT='real: %e  user: %U  sys: %S  cpu: %P%%'
echo "DOLFIN version: $VERSION ($DATE)"
echo "Compiler flags: $CFLAGS"
uname -a
`../../../../config/dolfin-config --compiler` --version

# Non-stiff
run 100 0
mv timings.m timings_M100_b0.m
run 200 0
mv timings.m timings_M200_b0.m

# Stiff
run 100 100
mv timings.m timings_M100_b100.m
run 200 100
mv timings.m timings_M200_b100.m
