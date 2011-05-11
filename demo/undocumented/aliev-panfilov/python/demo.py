# Copyright (C) 2007 Kristian B. Oelgaard
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
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with DOLFIN.  If not, see <http://www.gnu.org/licenses/>.
#
# Original implementation: ../cpp/main.cpp by Anders Logg.
#
# First added:  2007-11-15
# Last changed: 2007-11-28
#
# This demo solves a simple model for cardiac excitation,
# proposed in a 1995 paper by Aliev and Panfilov.

from dolfin import *

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

ode = AlievPanfilov(2, 300)
ode.solve()
