#!/usr/bin/env python

# Copyright (C) 2008 Rolv Erlend Bredesen
#
# This file is part of DOLFIN.
#
# DOLFIN is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# DOLFIN is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with DOLFIN.  If not, see <http://www.gnu.org/licenses/>.
#
# Modified by Benjamin Kehlet
#
# First added:  2008-04-03
# Last changed: 2010-08-29

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

