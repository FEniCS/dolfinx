from dolfin import *
from math import *

# Plot result

from primal import *
from pylab import *

def plotComponent(i):
    clf()

    subplot(311)
    plot(t, u[:, i], label='u(i)')
    grid(True)
    title('Heat Equation - i = ' + str(i))
    ylabel('u')

    subplot(312)
    plot(t, r[:, i], label='r(i)')
    ylabel('r')

    subplot(313)
    semilogy(t, k[:, i], label='k(i)')
    ylabel('k')
    
    
    savefig('heat-' + str(i) + '.eps')

    #show()

# Plot a few components

i1 = 10
i2 = 20
i3 = 30
plotComponent(i1)
plotComponent(i2)
plotComponent(i3)

