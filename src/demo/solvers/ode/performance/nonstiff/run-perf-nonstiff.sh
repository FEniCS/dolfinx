#!/bin/sh

function run()
{
/usr/bin/time -f"$TIMEFORMAT" -otlog ./dolfin-ode-perf-nonstiff $1 $2 > /dev/null
TIME=$( cat tlog )
echo "Elapsed time:   " $TIME
echo -n $( echo $TIME | awk '{ print $4 }' ) " + " $( echo $TIME | awk '{ print $6 }' ) >> timings.m
echo -n " " >> timings.m
echo
}


VERSION=`../../../../../config/dolfin-config --version`
DATE=`date`
CFLAGS=`../../../../../config/dolfin-config --cflags`
TIMEFORMAT='real: %e  user: %U  sys: %S  cpu: %P%%'

echo "DOLFIN version: $VERSION ($DATE)"
echo "Compiler flags: $CFLAGS"

uname -a
`../../../../../config/dolfin-config --compiler` --version
echo "---------------------------------------------------------------------------"

N="4 8 16 32 100 200 400 1000 2000 4000 10000"
#N="100 200 400"
#N="4 8 16"

echo "n = [ " $N "]" > timings.m


echo -n "t_mcg1 = [" >> timings.m

for i in $N
do

echo "mcG(1) n = " $i
run $i mcg

done

echo

echo "]" >> timings.m


echo -n "t_mdg0 = [" >> timings.m

for i in $N
do

echo "mdG(0) n = " $i
run $i mdg

done

echo "]" >> timings.m

echo


#echo -n "t_cg1 = [" >> timings.m

#for i in $N
#do

#echo "cG(1) n = " $i
#run $i cg

#done

#echo "]" >> timings.m

#echo

#echo -n "t_dg0 = [" >> timings.m

#for i in $N
#do

#echo "dG(0) n = " $i
#run $i dg

#done

#echo "]" >> timings.m

echo "---------------------------------------------------------------------------"
