#!/bin/bash
#
# Run from test/unit/python/ as:
#  ./scripts/mpidebug.sh <additional pytest args>
#
# Configure number of repeats and number of mpi processes like this:
#  REPEAT=2 PROCS='3 4' ./scripts/mpidebug.sh fem book
#
# This script runs pytest in with mpi such that output from each process
# is visible in separate xterm windows, and on a crash you can abort
# and exit into gdb to inspect the stacktrace in each xterm separately.
#
# You can pass a list of test files or test modules as arguments to this script.
# These tests are all executed in a single pytest instance.
# See also splitmpidebug.sh.
#

# ... PYTESTARGS = arguments passed directly to pytest (default .)
PYTESTARGS=${@:-.}

# ... Max number of times to run each test file
REPEAT=${REPEAT:-1}

# ... List of number of mpi processes to try
PROCS=${PROCS:-3}

echo
echo Running $REPEAT times with $PROCS processes with pytest args
echo $PYTESTARGS
echo

for p in $PROCS
do
    n=1
    # Loop at most $m times until file fails
    while [ $? -eq 0 -a  $n -le $REPEAT ]
    do
        echo === Take $n, $p processes, `date`: $f
        n=$((n+1))
        # Clean before each run to remove error sources from file system
        ./scripts/clean.sh
        # Run!
        mpirun -n $p xterm -e gdb -ex r -ex q -args python -B -m pytest -sv $PYTESTARGS
    done
done
