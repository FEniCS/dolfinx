#!/bin/sh

cd ../../
TOPLEVEL=`pwd`

# Create symbolic links

ln -s $TOPLEVEL/src/utils/octave/pdeplot.m src/demo/solvers/convdiff/
ln -s $TOPLEVEL/src/utils/octave/pdemesh.m src/demo/solvers/convdiff/
ln -s $TOPLEVEL/src/utils/octave/pdesurf.m src/demo/solvers/convdiff/

ln -s $TOPLEVEL/src/utils/matlab/plotslab.m src/demo/solvers/ode/test/
