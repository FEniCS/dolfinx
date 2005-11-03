from dolfin import *
from math import *

# Plot result

from primal import *
from pylab import *

import time

x = arange(0, 5.0, 5.0 / (size(u, 1) - 1))

def createPlots():
    plots = [0, 0, 0]

    subplot(311)
    plots[0], = plot(x, u[0, :], label='u(i)', hold=True)
    grid(True)
    title('Reaction')
    ylabel('u')

    subplot(312)
    plots[1], = plot(x, r[0, :], label='r(i)', hold=True)
    ylabel('r')

    subplot(313)
    plots[2], = semilogy(x, k[0, :], label='k(i)', hold=True)
    ylabel('k')
    
    return plots


def plotComponent(plots, i):
    plots[0].set_ydata(u[i, :])
    plots[1].set_ydata(r[i, :])
    plots[2].set_ydata(k[i, :])
    
    draw()

ion()

plots = createPlots()
while(True):
    for i in range(0, size(t, 0)):
        plotComponent(plots, i)
