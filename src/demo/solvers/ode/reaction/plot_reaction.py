from dolfin import *
from math import *
import MA

# Plot result

from primal import *

# Local error estimate
e_est = k * r

from pylab import *

import time

x = arange(0, 5.0, 5.0 / (size(u, 1) - 1))

def createPlots():
    plots = [0, 0, 0, 0]

    xmin = 0
    xmax = 5

    umax = MA.maximum(abs(u))

    subplot(411)
    plots[0], = plot(x, u[0, :], label='u(i)', hold=True)
    grid(True)
    title('Reaction')
    ylabel('u')
    xlim([xmin, xmax])
    ylim([-1.5 * umax, 1.5 * umax])

    rmax = MA.maximum(abs(r))

    subplot(412)
    plots[1], = plot(x, r[0, :], label='r(i)', hold=True)
    ylabel('r')
    xlim([xmin, xmax])
    ylim([-1.5 * rmax, 1.5 * rmax])

    e_estmax = MA.maximum(abs(e_est))

    subplot(413)
    plots[2], = plot(x, e_est[0, :], label='e_est(i)', hold=True)
    ylabel('e_est')
    xlim([xmin, xmax])
    ylim([-1.5 * e_estmax, 1.5 * e_estmax])

    kmax = MA.maximum(k)
    kmin = MA.minimum(k)

    subplot(414)
    plots[3], = semilogy(x, k[0, :], label='k(i)', hold=True)
    ylabel('k')
    ylim([kmin, kmax])
    xlim([xmin, xmax])

    
    return plots


def plotComponent(plots, i):
    plots[0].set_ydata(u[i, :])
    plots[1].set_ydata(r[i, :])
    plots[2].set_ydata(e_est[i, :])
    plots[3].set_ydata(k[i, :])
    
    draw()

ion()

plots = createPlots()
while(True):
    for i in range(0, size(t, 0)):
        plotComponent(plots, i)
