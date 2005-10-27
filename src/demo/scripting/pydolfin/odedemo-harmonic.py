from dolfin import *
from math import *

class Harmonic(ODE):
    def __init__(self):
        print "Harmonic ctor"
        ODE.__init__(self, 2, 20.0 * pi)
        self.sparse()

    def u0(self, i):
        print "Evaluating Python u0 function: "
        if i == 0:
            return 0
        else:
            return 1

    def fmono(self, u, t, y):
        realArray_setitem(y, 0, realArray_getitem(u, 1))
        realArray_setitem(y, 1, -realArray_getitem(u, 0))

    def fmulti(self, u, t, i):
        if i == 0:
            return realArray_getitem(u, 1)
        else:
            return -realArray_getitem(u, 0)

dolfin_set("method", "mcg")
dolfin_set("order", 1)

dolfin_set("file name", "primal.py")
dolfin_set("number of samples", 1000)

dolfin_set("tolerance", 1e-4)
#dolfin_set("tolerance", 1e-8)

ode = Harmonic()
ode.solve()

# Plot result

from primal import *
from pylab import *

#gplt.plot(t, u[:, 0], 'title "u(0)" with lines',
#          t, u[:, 1], 'title "u(1)" with lines')
#gplt.title('Harmonic oscillator')
#gplt.xtitle("t")
#gplt.ytitle("u")

title('Harmonic oscillator')
subplot(311)
plot(t, u[:, 0], label='u(0)')
plot(t, u[:, 1], label='u(1)')
grid(True)
title('Harmonic oscillator')
ylabel('u')

subplot(312)
plot(t, r[:, 0], label='r(0)')
plot(t, r[:, 1], label='r(1)')
ylabel('r')

subplot(313)
plot(t, k[:, 0], label='k(0)')
plot(t, k[:, 1], label='k(1)')
ylabel('k')
show()
