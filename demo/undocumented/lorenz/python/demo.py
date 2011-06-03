"Lorenz demo"

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
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with DOLFIN. If not, see <http://www.gnu.org/licenses/>.
#
# Modified by Anders Logg, 2008.
# Modified by Benjamin Kehlet, 2010
#
# First added:  2008-04-03
# Last changed: 2010-08-29

import numpy
from dolfin import *

class Lorenz(ODE):

    def __init__(self, N=3, T=50.):
        ODE.__init__(self, N, T)

        # Parameters
        self.s = 10.0;
        self.b = 8.0 / 3.0;
        self.r = 28.0;
 

    def u0(self, u):
        u[0] = 1.0;
        u[1] = 0.0;
        u[2] = 0.0;

    def f(self, u, t, y):
        y[0] = self.s*(u[1] - u[0]);
        y[1] = self.r*u[0] - u[1] - u[0]*u[2];
        y[2] = u[0]*u[1] - self.b*u[2];

    def J(self, x, y, u, t):
        y[0] = self.s*(x[1] - x[0]);
        y[1] = (self.r - u[2])*x[0] - x[1] - u[0]*x[2];
        y[2] = u[1]*x[0] + u[0]*x[1] - self.b*x[2];

def myplot():
    import pylab
    from mpl_toolkits.mplot3d import Axes3D

    pylab.ion()

    r = numpy.fromfile('solution_r.data', sep=' ')
    r.shape = len(r)//3, 3
    print "Residual in l2 norm:", pylab.norm(r)

    u = numpy.fromfile('solution_u.data', sep=' ')
    u.shape = len(u)//3, 3

    fig = pylab.figure()
    ax = Axes3D(fig)

    ax.plot(u[:,0], u[:,1], u[:,2], linewidth=0.25)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_title('Lorenz attractor')
    pylab.show()


lorenz = Lorenz(T=50)

lorenz.parameters["number_of_samples"] = 5000
lorenz.parameters["initial_time_step"] = 0.01
lorenz.parameters["fixed_time_step"] = True
lorenz.parameters["method"] = "cg"
lorenz.parameters["order"] = 5
lorenz.parameters["discrete_tolerance"] = 1e-10
lorenz.parameters["save_solution"] = True


lorenz.solve();

myplot()
