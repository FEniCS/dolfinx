# Copyright (C) 2005 Anders Logg.
# Licensed under the GNU GPL Version 2.
#
# Run benchmarks and collect results

TOLERANCES='1e-1 5e-2 1e-2 5e-3 1e-3 5e-4 1e-4 5e-5 1e-5 5e-6 1e-6'
#TOLERANCES='1e-1'
LOGFILE='output.log'
RESULTS='bench.log'

rm -f $RESULTS
date >> $RESULTS
echo '' >> $RESULTS
echo -e "Method \t TOL \t Error \t\t CPU time \t Steps \t Iterations \t Index" > $RESULTS
echo '------------------------------------------------------------------------------' >> $RESULTS

for TOL in $TOLERANCES; do
    
    echo "Mono-adaptive cG(1), TOL = $TOL ..."
    rm -f $LOGFILE
    ./dolfin-ode-reaction cg $TOL > $LOGFILE

    if [ 'x'`grep 'did not converge' $LOGFILE` = 'x' ]; then
	CPUTIME=`cat $LOGFILE | grep 'Solution computed in' | awk '{ print $4 }'`
	STEPS=`cat $LOGFILE | grep 'Total number of (macro) time steps' | awk '{ print $7 }'`
	ITERATIONS=`cat $LOGFILE | grep 'Average number of global iterations' | awk '{ print $8 }'`
	ERROR=`octave checkerror.m | grep Error | awk '{ print $2 }'`
	
	echo -e "cG(1) \t $TOL \t $ERROR \t $CPUTIME \t $STEPS \t $ITERATIONS" >> $RESULTS
    else
	echo -e "cG(1) \t $TOL \t Did not converge" >> $RESULTS
    fi

done

echo '' >> $RESULTS

for TOL in $TOLERANCES; do
    
    echo "Multi-adaptive mcG(1), TOL = $TOL ..."
    rm -f $LOGFILE
    ./dolfin-ode-reaction mcg $TOL > $LOGFILE

    if [ 'x'`grep 'did not converge' $LOGFILE` = 'x' ]; then
	CPUTIME=`cat $LOGFILE | grep 'Solution computed in' | awk '{ print $4 }'`
	STEPS=`cat $LOGFILE | grep 'Total number of (macro) time steps' | awk '{ print $7 }'`
	ITERATIONS=`cat $LOGFILE | grep 'Average number of global iterations' | awk '{ print $8 }'`
	ERROR=`octave checkerror.m | grep Error | awk '{ print $2 }'`
	INDEX=`cat $LOGFILE | grep 'Multi-adaptive efficiency index' | awk '{ print $4 }'`

	echo -e "mcG(1) \t $TOL \t $ERROR \t $CPUTIME \t $STEPS \t $ITERATIONS \t $INDEX" >> $RESULTS
    else
	echo -e "mcG(1) \t $TOL \t Did not converge" >> $RESULTS
    fi

done

echo ''
cat $RESULTS
