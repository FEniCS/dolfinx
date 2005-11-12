# Copyright (C) 2005 Anders Logg.
# Licensed under the GNU GPL Version 2.
#
# Run benchmarks and collect results

#TOLERANCES='1e-3 5e-4 1e-4 5e-5 1e-5'
#TOLERANCES='1e-3 5e-4 1e-4 5e-5 1e-5'
#TOLERANCES='1e-1 5e-2 1e-2 5e-3 1e-3 5e-4 1e-4 5e-5 1e-5 5e-6 1e-6'
#TOLERANCES='1e-1'
LOGFILE='output.log'
RESULTS='bench.log'

function run_set()
{
    METHOD=$1
    SOLVER=$2
    TOLMAX=$3
    K0=$4
    KMAX=$5
    GAMMA=$6
    dolfin_info("method - 'cg' or 'mcg'");
    dolfin_info("solver - 'fixed-point' or 'newton'");
    dolfin_info("TOL    - tolerance");
    dolfin_info("k0     - initial time step");
    dolfin_info("kmax   - initial time step");
    dolfin_info("gamma  - reaction rate, something like 100.0 or 1000.0");    

    METHOD_SHORT=$1
    METHOD_LONG=$2
    TOLMAX=$3

    rm -f $RESULTS
    date >> $RESULTS
    echo '' >> $RESULTS
    echo -e "Method \t TOL \t Error \t\t CPU time \t Steps \t\t Iterations \t Index" > $RESULTS
    echo '------------------------------------------------------------------------------' >> $RESULTS
    
    FRACTIONS='1.0 0.5 0.1 0.05'
    for FRACTION in $FRACTIONS; do
	
	TOL=`echo $FRACTION $TOLMAX | awk '{ print $1*$2 }'`
	echo "$METHOD_LONG, TOL = $TOL ..."
	rm -f $LOGFILE
	./dolfin-ode-reaction cg $TOL > $LOGFILE

	if [ 'x'`grep 'Solution computed in' $LOGFILE` = 'x' ]; then
	    echo -e "$METHOD_SHORT \t $TOL \t Unable to solve" >> $RESULTS
	else
	    CPUTIME=`cat $LOGFILE | grep 'Solution computed in' | awk '{ print $4 }'`
	    STEPS=`cat $LOGFILE | grep 'Total number of (macro) time steps' | awk '{ print $7 }'`
	    ITERATIONS=`cat $LOGFILE | grep 'Average number of global iterations' | awk '{ print $8 }'`
	    REJECTED=`cat $LOGFILE | grep 'Number of rejected time steps' | awk '{ print $6 }'`
	    ERROR=`octave checkerror.m | grep Error | awk '{ print $2 }'`
	    
	    echo -e "cG(1) \t $TOL \t $ERROR \t $CPUTIME \t $STEPS ($REJECTED) \t $ITERATIONS" >> $RESULTS
	fi
	
    done

}


run_set 'cG(1)' 'Multi-adaptive cG(1)' 1e-3

echo ""
cat $RESULTS
