#!/bin/sh

cd ../../
TOPLEVEL=`pwd`

# Function for creating symbolic links to Octave scripts
link_octave() {
    ln -sf $TOPLEVEL/src/utils/octave/pdeplot.m $1
    ln -sf $TOPLEVEL/src/utils/octave/pdemesh.m $1
    ln -sf $TOPLEVEL/src/utils/octave/pdesurf.m $1
}

# Function for creating symbolic links to OpenDX programs
link_dx() {
    ln -sf $TOPLEVEL/src/utils/opendx/dolfin.net $1
    ln -sf $TOPLEVEL/src/utils/opendx/dolfin.cfg $1
}

# Function for creating symbolic link to plotslab.m
link_plotslab() {
    ln -sf $TOPLEVEL/src/utils/matlab/plotslab.m $1
}

# Create links to Octave scripts
link_octave src/demo/solvers/convdiff/
link_octave src/demo/solvers/wave/
link_octave src/demo/solvers/wave-vector/
link_octave src/demo/solvers/elasticity/
link_octave src/demo/solvers/elasticity-stationary/

# Creat links to OpenDX programs
link_dx src/demo/solvers/ode/bistable/

# Create links to plotslab.m
link_plotslab src/demo/solvers/ode/test/
link_plotslab src/demo/solvers/ode/stiff/testproblems/
