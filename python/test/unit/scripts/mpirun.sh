#!/bin/bash
#
# NB! Run as e.g.
#  cd test/unit/python
#  ./scripts/mpirun.sh <pytest args>
#
./scripts/clean.sh
mpirun -n 3 python -B -m pytest -v $@
