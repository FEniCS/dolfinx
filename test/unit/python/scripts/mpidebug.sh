#!/bin/bash
#
# NB! Run as e.g.
#  ./scripts/mpidebug.sh io/python/
#
echo; while [ $? -eq 0 ]; do ./scripts/clean.sh && mpirun -n 3 xterm -e gdb -ex r -ex q -args python -m pytest -sv $@; done
