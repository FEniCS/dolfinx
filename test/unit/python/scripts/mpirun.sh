#!/bin/bash
#
# NB! Run as e.g.
#  ./scripts/mpirun.sh io/python/
#
./scripts/clean.sh && mpirun -n 3 python -m pytest $@
