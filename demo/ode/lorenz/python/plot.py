#!/usr/bin/env python

__author__ = "Rolv Erlend Bredesen <rolv@simula.no>"
__date__ = "2008-04-03 -- 2008-04-03"
__copyright__ = "Copyright (C) 2008 Rolv Erlend Bredesen"
__license__  = "GNU LGPL Version 2.1"

import numpy 
from pylab import *
import matplotlib.axes3d as p3
from solution import u, t, k, r

print "Residual:", norm(r)

ion() # interactive on
plot(t, u)
legend(('x', 'y', 'z'))
xlabel('t')
ylabel('U(t)')
grid('on')

x, y, z = numpy.hsplit(u, 3)

ax = p3.Axes3D(figure())
ax.plot3d(x,y,z)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
savefig('lorenz.png')
draw()

figure()
plot(t, x, label='x')
legend()

