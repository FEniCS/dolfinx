#!/bin/bash
#
# NB! Run as e.g.
#  ./scripts/prun.sh io/python/
#
./scripts/clean.sh && python -B -m pytest -n 4 -v $@
