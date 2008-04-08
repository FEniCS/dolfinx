#!/usr/bin/env python
"""
Lorenz demo
"""

__author__ = "Rolv Erlend Bredesen <rolv@simula.no>"
__date__ = "2008-04-03 -- 2008-04-03"
__copyright__ = "Copyright (C) 2008 Rolv Erlend Bredesen"
__license__  = "GNU LGPL Version 2.1"

from numpy import empty
from dolfin import *
  
class Lorenz(ODE):
    
    # Parameters
    s = 10.0;
    b = 8.0 / 3.0;
    r = 28.0;
        
    def __init__(self, N=3, T=50.):
        ODE.__init__(self, N, T)

        # Work arrays corresponding to uBlasVectors
        self.u = empty(N)
        self.x = empty(N)
        self.y = empty(N)
        
    def u0(self, u_):
        u = self.u
        u[0] = 1.0;
        u[1] = 0.0;
        u[2] = 0.0;
        u_.set(u)
        
    def f(self, u_, t_, y_):
        u = self.u
        u_.get(u)
        y = self.y
        y[0] = self.s*(u[1] - u[0]);
        y[1] = self.r*u[0] - u[1] - u[0]*u[2];
        y[2] = u[0]*u[1] - self.b*u[2];
        y_.set(y)
        
    def J(self, x_, y_, u_, t):
        x = self.x
        y = self.y
        u = self.u
        u_.get(u)
        x_.get(x)
        y[0] = self.s*[x[1] - x[0]];
        y[1] = [self.r - u[2]]*x[0] - x[1] - u[0]*x[2];
        y[2] = u[1]*x[0] + u[0]*x[1] - self.b*x[2];
        y_.set(y)
        
dolfin_set("ODE number of samples", 500);
dolfin_set("ODE initial time step", 0.01);
#dolfin_set("ODE fixed time step", True);
#dolfin_set("ODE nonlinear solver", "newton");
dolfin_set("ODE method", "cg");
dolfin_set("ODE order", 5); # 1 in exersize
dolfin_set("ODE discrete tolerance", 1e-12);
#dolfin_set("ODE save solution", 1);

lorenz = Lorenz()
lorenz.solve();
