#!/bin/sh
#
# Copyright (C) 2004 Johan Jansson.
# Licensed under the GNU GPL Version 2.
#
# Modified by Anders Logg, 2004.

function runsingle()
{
    time ./dolfin-ode-perf-test $1 $2 $3 $4 1> log1 2> tlog
    TIME=$( cat tlog )
    echo "Elapsed time:   " $TIME
    echo -n $( echo $TIME | awk '{ print $4 }' ) " + " $( echo $TIME | awk '{ print $6 }' ) >> timings.m
    echo -n " " >> timings.m
    echo
}

function run()
{
    VERSION=`../../../../config/dolfin-config --version`
    DATE=`date`
    CFLAGS=`../../../../config/dolfin-config --cflags`
    TIMEFORMAT='real: %3R  user: %3U  sys: %3S  cpu: %P%%'
    
    echo "DOLFIN version: $VERSION ($DATE)"
    echo "Compiler flags: $CFLAGS"
    
    uname -a
    `../../../../config/dolfin-config --compiler` --version
    echo "---------------------------------------------------------------------------"

    #N="10 20 40 100 200 400 1000 2000 4000 10000 20000 50000 100000"
    N="10 20 40 100 200 400 1000"
    M="100"
    B="100"
    
    if [ "$1" = "" ]; then
	M=100
    else
	M=$1	
    fi
    
    if [ "$2" = "" ]; then
	B=100
    else
	B=$2
    fi
    
    echo "M = "$M
    echo "B = "    
    echo
    #N="100 200 400"
    #N="4 8 16"    
    echo "n = [ " $N "];" > timings.m
    echo -n "t_mdg0 = [" >> timings.m

    for i in $N; do
	
	echo "mdG(0) n = " $i
	runsingle $i $M $B mdg
	
    done
    
    echo "];" >> timings.m
    echo "k_mdg0 = " $( echo "timings; lsquares(n, t_mdg0)(1)" | octave -q | awk '{ print $3 }' ) ";" >> timings.m
    echo "m_mdg0 = " $( echo "timings; lsquares(n, t_mdg0)(2)" | octave -q | awk '{ print $3 }' ) ";" >> timings.m
    echo
    echo -n "t_dg0 = [" >> timings.m
    
    for i in $N; do
	
	echo "dG(0) n = " $i
	runsingle $i $M mono
	
    done
    
    echo "];" >> timings.m
    echo "k_dg0 = " $( echo "timings; lsquares(n, t_mdg0)(1)" | octave -q | awk '{ print $3 }' ) ";" >> timings.m
    echo "m_dg0 = " $( echo "timings; lsquares(n, t_mdg0)(2)" | octave -q | awk '{ print $3 }' ) ";" >> timings.m
    echo
    echo "---------------------------------------------------------------------------"

}

run 100 100
mv timings.m timings_m100_b100.m
run 200 100
mv timings.m timings_m200_b100.m
