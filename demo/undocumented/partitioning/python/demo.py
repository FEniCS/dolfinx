"""Run this demo in parallel by

    mpirun -n <n> python demo.py

where <n> is the desired number of processes."""

__author__ = "Ola Skavhaug (skavhaug@simula.no) and Anders Logg (logg@simula.no)"
__date__ = "2008-12-18 -- 2008-12-18"
__copyright__ = "Copyright (C) 2008 Ola Skavhaug and Anders Logg"
__license__  = "GNU LGPL Version 2.1"

from dolfin import *

# Read in mesh from XML file in parallel
mesh = Mesh("unitsquare.xml.gz")

# Plot partition
plot(mesh, interactive=True)
