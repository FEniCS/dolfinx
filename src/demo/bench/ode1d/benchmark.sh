#!/bin/sh
#
# Copyright (C) 2005 Johan Jansson.
# Licensed under the GNU GPL Version 2.
#
# Note: /usr/bin/time (Debian package time) is needed
# to avoid using the bash built-in time, which is
# difficult to redirect.

function runsingle()
{
    /usr/bin/time -f"$TIMEFORMAT" -otlog ./dolfin-bench-ode $1 $2 $3 $4 $5 1> log1 2> log2
    TIME=$( cat tlog )
    echo "Elapsed time: "$TIME
    echo -n $( echo $TIME | awk '{ print $4 }' ) " + " $( echo $TIME | awk '{ print $6 }' ) >> timings.m
    echo -n " " >> timings.m
    echo
}

function run()
{
    n="1000 2000 4000 8000"

    echo "---------------------------------------------------------------------------"    
    echo "n = [ " $n "];" > timings.m
    echo -n "t1 = [" >> timings.m

    # Run multi-adaptive test
    echo "Running multi-adaptive performance test"
    echo "---------------------------------------"
    echo

    for i in $n; do
	echo "mcG(1) n = " $i
	runsingle mcg 1 $i 1e-10 1e-6	
    done

    echo "];" >> timings.m

    echo
    echo -n "t2 = [" >> timings.m

    # Run mono-adaptive test
    echo "Running mono-adaptive performance test"
    echo "--------------------------------------"
    echo

    for i in $n; do
	echo "cG(1) n = " $i
	runsingle cg 1 $i 1e-10 1e-6	
    done

    echo "];" >> timings.m

}

VERSION=`../../../config/dolfin-config --version`
DATE=`date`
CFLAGS=`../../../config/dolfin-config --cflags`
TIMEFORMAT='real: %e  user: %U  sys: %S  cpu: %P%%'
echo "DOLFIN version: $VERSION ($DATE)"
echo "Compiler flags: $CFLAGS"
uname -a
`../../../config/dolfin-config --compiler` --version

run

