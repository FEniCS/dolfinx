#!/bin/sh

cd ../../
TOPLEVEL=`pwd`

# Create symbolic links

ln -sf $TOPLEVEL/src/utils/octave/pdeplot.m src/demo/solvers/convdiff/
ln -sf $TOPLEVEL/src/utils/octave/pdemesh.m src/demo/solvers/convdiff/
ln -sf $TOPLEVEL/src/utils/octave/pdesurf.m src/demo/solvers/convdiff/

ln -sf $TOPLEVEL/src/utils/matlab/plotslab.m src/demo/solvers/ode/test/
