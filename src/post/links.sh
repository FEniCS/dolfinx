#!/bin/sh

cd ../../
TOPLEVEL=`pwd`

# Create symbolic links
# Make a sub function for linking the Octave scripts

ln -sf $TOPLEVEL/src/utils/octave/pdeplot.m src/demo/solvers/convdiff/
ln -sf $TOPLEVEL/src/utils/octave/pdemesh.m src/demo/solvers/convdiff/
ln -sf $TOPLEVEL/src/utils/octave/pdesurf.m src/demo/solvers/convdiff/

ln -sf $TOPLEVEL/src/utils/octave/pdeplot.m src/demo/solvers/wave/
ln -sf $TOPLEVEL/src/utils/octave/pdemesh.m src/demo/solvers/wave/
ln -sf $TOPLEVEL/src/utils/octave/pdesurf.m src/demo/solvers/wave/

ln -sf $TOPLEVEL/src/utils/octave/pdeplot.m src/demo/solvers/wave-vector/
ln -sf $TOPLEVEL/src/utils/octave/pdemesh.m src/demo/solvers/wave-vector/
ln -sf $TOPLEVEL/src/utils/octave/pdesurf.m src/demo/solvers/wave-vector/

ln -sf $TOPLEVEL/src/utils/octave/pdeplot.m src/demo/solvers/elasticity/
ln -sf $TOPLEVEL/src/utils/octave/pdemesh.m src/demo/solvers/elasticity/
ln -sf $TOPLEVEL/src/utils/octave/pdesurf.m src/demo/solvers/elasticity/

ln -sf $TOPLEVEL/src/utils/octave/pdeplot.m src/demo/solvers/elasticity-stationary/
ln -sf $TOPLEVEL/src/utils/octave/pdemesh.m src/demo/solvers/elasticity-stationary/
ln -sf $TOPLEVEL/src/utils/octave/pdesurf.m src/demo/solvers/elasticity-stationary/

ln -sf $TOPLEVEL/src/utils/matlab/plotslab.m src/demo/solvers/ode/test/
ln -sf $TOPLEVEL/src/utils/matlab/plotslab.m src/demo/solvers/ode/stiff/testproblems/
