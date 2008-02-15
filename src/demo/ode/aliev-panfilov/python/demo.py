# This demo solves a simple model for cardiac excitation,
# proposed in a 1995 paper by Aliev and Panfilov.
#
# Original implementation: ../cpp/main.cpp by Anders Logg.
#
__author__ = "Kristian B. Oelgaard (k.b.oelgaard@tudelft.nl)"
__date__ = "2007-11-15 -- 2007-11-28"
__copyright__ = "Copyright (C) 2007 Kristian B. Oelgaard"
__license__  = "GNU LGPL Version 2.1"

from dolfin import *

# FIXME: Not working, see notice below
import sys
print "This demo is not working, please fix me"
sys.exit(1)

class AlievPanfilov(ODE):
    def __init__(self, N, T):
        ODE.__init__(self, N, T)
        # Set parameters
        self.a = 0.15
        self.eps0 = 0.002
        self.k    = 8.0
        self.mu1  = 0.07
        self.mu2  = 0.3
  
    def u0(self, u):
        u[0] = 0.2
        u[1] = 0.0

    def f(self, u, t, y):
        eps = self.eps0 + self.mu1*u[1] / (u[0] + self.mu2)
        y[0] = -self.k*u[0]*(u[0] - self.a)*(u[0] - 1.0) - u[0]*u[1]
        y[1] = eps*(-u[1] - self.k*u[0]*(u[0] - self.a - 1.0))


ode = ODE(2, 300)
# ERROR:
# Traceback (most recent call last):
#   File "demo.py", line 39, in <module>
#     ode = ODE(2, 300)
# TypeError: __init__() takes exactly 1 argument (3 given)

#ode = ODE()
# ERROR:
# Traceback (most recent call last):
#   File "demo.py", line 37, in <module>
#     ode = ODE()
#   File "/home/oelgaard/fenics/dolfin/local/lib/python2.5/site-packages/dolfin/dolfin.py", line 6208, in __init__
#     def __init__(self): raise AttributeError, "No constructor defined"
# AttributeError: No constructor defined


#ode = AlievPanfilov(2, 300)
#  ode.solve();


