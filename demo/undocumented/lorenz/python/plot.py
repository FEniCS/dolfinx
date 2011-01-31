#!/usr/bin/env python

__author__    = "Rolv Erlend Bredesen <rolv@simula.no>"
__date__      = "2008-04-03 -- 2010-08-29"
__copyright__ = "Copyright (C) 2008 Rolv Erlend Bredesen"
__license__   = "GNU LGPL Version 2.1"

# Modified by Benjamin Kehlet

import numpy
from pylab import *
from mpl_toolkits.mplot3d import Axes3D
from solution import u, t, k, r

print "Residual:", norm(r)

ion() # interactive on

# Plot all three components
plot(t, u)
legend(('x', 'y', 'z'))
xlabel('t')
ylabel('U(t)')
grid('on')

# Trajectory plot
ax = Axes3D(figure())
ax.plot(u[:,0] ,u[:,1], u[:,2])
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')

# Plot only x component
figure()
plot(t, u[:,0], label='x')
legend()

show()

