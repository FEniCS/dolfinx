#!/bin/sh

function run()
{
/usr/bin/time -f"$TIMEFORMAT" -otlog ./dolfin-ode-perf-stiff $1 $2 $3 $4 1> log1 2> log2
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
echo "B = "$B


#N="100 200 400"
#N="4 8 16"

echo "n = [ " $N "];" > timings.m


#echo -n "t_mcg1 = [" >> timings.m

#for i in $N
#do

#echo "mcG(1) n = " $i
#run $i mcg

#done

#echo

#echo "]" >> timings.m


echo -n "t_mdg0 = [" >> timings.m

for i in $N
do

echo "mdG(0) n = " $i
run $i $M $B mdg

done

echo "];" >> timings.m

echo "k_mdg0 = " $( echo "timings; lsquares(n, t_mdg0)(1)" | octave -q | awk '{ print $3 }' ) ";" >> timings.m

echo "m_mdg0 = " $( echo "timings; lsquares(n, t_mdg0)(2)" | octave -q | awk '{ print $3 }' ) ";" >> timings.m

#echo

#echo -n "t_dg0 = [" >> timings.m

#for i in $N
#do

#echo "dG(0) n = " $i
#run $i $M mono

#done

#echo "]" >> timings.m

#echo


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
