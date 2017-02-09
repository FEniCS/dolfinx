# Copyright (C) 2017 Simon Funke
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
# First added:  2017-02-09
# Last changed: 2017-02-09
#
# This demo program solves Poisson's equation on a domain defined by
# three overlapping and non-matching meshes. The solution is computed
# on a sequence of rotating meshes to test the multimesh
# functionality.

import argparse
import numpy 
from dolfin import *

parser = argparse.ArgumentParser()
parser.add_argument('num_parts', type=int, help='number of meshes', default=10)
parser.add_argument('--N_x', type=int, help='number of mesh divisions (mesh size)', default=2, required=False)
parser.add_argument('--random_seed', type=float, help='seed for random number generator for creating multimesh', default=None, required=False)
args = parser.parse_args()

print("Number of meshes: {}.".format(args.num_parts))
print("Number of mesh devisions: {}.".format(args.N_x))
print("Seed for random number generator for creating multimesh: {}.".format(args.random_seed))


def build_multimesh(num_parts, N_x):

    # Build multimesh
    multimesh = MultiMesh()
    h = 1/N_x

    # Add background mesh
    mesh = UnitSquareMesh(N_x, N_x)
    multimesh.add(mesh)

    # Add N-1 random sized and rotated rectangular meshes 
    for _ in range(num_parts-1):

        x0, x1 = numpy.sort(numpy.random.rand(2))
        y0, y1 = numpy.sort(numpy.random.rand(2))
        if x0 - x1 < DOLFIN_EPS:
            x1 += DOLFIN_EPS
        if y0 - y1 < DOLFIN_EPS:
            x1 += DOLFIN_EPS

        print "Add new rectanble mesh ({:.3f}, {:.3f}) x ({:.3f}, {:.3f}).".format(x0, y0, x1, y1)
        mesh = RectangleMesh(Point(x0, x1), Point(y0, y1), 
                             int(max((x1-x0)*N_x, 1)), int(max((y1-y0)*N_x, 1)))

        #mesh.rotate(numpy.random.rand()*180)
        multimesh.add(mesh)

    multimesh.build()
    return multimesh

if __name__ == "__main__":

    if MPI.size(mpi_comm_world()) > 1:
        info("Sorry, this demo does not (yet) run in parallel.")
        exit(0)

    # Build multimesh
    multimesh = build_multimesh(args.num_parts, args.N_x)

    # Assemble linear system
    # FIXME: This does not work yet
    #vol = assemble_multimesh(1*dx(domain=multimesh))

    # Alternative volume calculation
    V = MultiMeshFunctionSpace(multimesh, "Lagrange", 1)
    f = MultiMeshFunction(V)
    f.vector()[:] = 1.0
    vol = assemble_multimesh(f*dX)
    
    print "Computed volume: {}.".format(vol)
    print "Error: {}.".format(abs(1-vol))
    assert abs(vol-1) < 10e-10
