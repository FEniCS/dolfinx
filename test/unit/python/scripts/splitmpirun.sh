#!/bin/bash
#
# NB! Run as e.g.
#  cd test/unit/python
#  ./scripts/splitmpirun.sh <additional pytest args>
#
# This script runs each test*.py file with pytest from python under gdb in an xterm via mpirun in three processes.
# Dizzy? The point is this:
# - run this and wait
# - if it deadlocks, you can go into gdb in each subprocess:
#    - press Ctrl+C in each of the 3 xterms and follow instructions
#    - now you can inspect where each of the processes are stuck, probably in different MPI calls
#    - make a note of which test file you ran, now you know a single file that can fail
#

# ... Max number of times to run each test file
m=1

# ... FILES = test files or modules to try
# All test files separately:
FILES=*/test_*.py
# All modules separately:
#FILES=*
# Commandline args:
#FILES=$@


echo
echo Running $m times each of
echo $FILES
echo

for f in $FILES
do
    for p in 3
    do
        n=1
        # Loop at most $m times, continue even if file fails
        #while [ $n -lt $m ]
        # Loop at most $m times until file fails
        while [ $? -eq 0 -a  $n -le $m ]
        do
            echo === Take $n, `date`: $f
            n=$((n+1))
            # Clean before each run to remove error sources from file system
            ./scripts/clean.sh
            # Run!
            mpirun -np $p python -B -m pytest -sv $f
        done
    done
done
