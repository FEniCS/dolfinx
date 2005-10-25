from dolfin import *
from math import *

class Harmonic(ODE):
    def __init__(self):
        print "Harmonic ctor"
        ODE.__init__(self, 2, 4.0 * pi)
        self.sparse()

    def u0(self, i):
        print "Evaluating Python u0 function: "
        if i == 0:
            return 0
        else:
            return 1

    def fmono(self, u, t, y):
        print "Evaluating Python fmono function: "
        realArray_setitem(y, 0, realArray_getitem(u, 1))
        realArray_setitem(y, 1, -realArray_getitem(u, 0))

    def fmulti(self, u, t, i):
        print "Evaluating Python fmulti function: "
        print "u: " + str(realArray_getitem(u,0))
        print "i: " + str(i)
        if i == 0:
            return realArray_getitem(u, 1)
        else:
            return -realArray_getitem(u, 0)

dolfin_set("method", "mcg")
dolfin_set("order", 1)
dolfin_set("file name", "primal.py")

ode = Harmonic()
ode.solve()

# Plot result

from primal import *

gplt.plot(t, u[:, 0], 'title "u(0)" with linespoints',
          t, u[:, 1], 'title "u(1)" with linespoints')
gplt.title('Harmonic oscillator')
gplt.xtitle("t")
gplt.ytitle("u")
