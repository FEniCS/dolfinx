from dolfin import *
from math import *

class Simple(ODE):
    def __init__(self):
        print "Simple ctor"
        ODE.__init__(self, 1)
        self.T = 1.0

    def u0(self, i):
        print "Evaluation Python u0 function: "
        return 0.0

    def f(self, u, t, i):
        print "Evaluation Python RHS function: "
        return 1.0
    
ode = Simple()
print "ODE size: " + str(ode.size())
ode.solve()
