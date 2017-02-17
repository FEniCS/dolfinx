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
# Modified by August Johansson 2017
#
# First added:  2017-02-09
# Last changed: 2017-02-18
#
# This demo program creates a layer of meshes over a unit square mesh and checks
# that the volume is 1.0.

import argparse
import numpy
from dolfin import *

parser = argparse.ArgumentParser()
parser.add_argument('num_parts', type=int, help='number of meshes', default=10)
parser.add_argument('--N_x', type=int, help='number of mesh divisions (mesh size)', default=2, required=False)
parser.add_argument('--random_seed', help='seed for random number generator for creating multimesh', default=1, required=False)
parser.add_argument('--no_random_rotation', action='store_true', help='Do not rotate each mesh with a random angle', default=False, required=False)
args = parser.parse_args()

print("Number of meshes: {}.".format(args.num_parts))
print("Number of mesh devisions: {}.".format(args.N_x))
print("Seed for random number generator for creating multimesh: {}.".format(args.random_seed))
print("Do not rotate each mesh with a random angle: {}.".format(args.no_random_rotation))
numpy.random.seed(args.random_seed)


def build_multimesh(num_parts, N_x, no_random_rotation):

    # Create multimesh
    multimesh = MultiMesh()

    # Add background mesh
    mesh = UnitSquareMesh(N_x, N_x)
    multimesh.add(mesh)

    # Add num_parts-1 random sized and rotated rectangular meshes
    while (multimesh.num_parts() < num_parts):

        x0, x1 = numpy.sort(numpy.random.rand(2))
        y0, y1 = numpy.sort(numpy.random.rand(2))
        if abs(x1 - x0) < DOLFIN_EPS:
            x1 += DOLFIN_EPS
        if abs(y1 - y0) < DOLFIN_EPS:
            y1 += DOLFIN_EPS

        mesh = RectangleMesh(Point(x0, y0), Point(x1, y1),
                             int(max(abs(x1-x0)*N_x, 1)), int(max(abs(y1-y0)*N_x, 1)))

        s =  "Add new rectangle mesh ({:.3f}, {:.3f}) x ({:.3f}, {:.3f}).".format(x0, y0, x1, y1)
        is_interior = True

        if not no_random_rotation:
            phi = numpy.random.rand()*180
            mesh.rotate(phi)
            coords = mesh.coordinates()
            is_interior = not numpy.any(coords < 0) and not numpy.any(coords > 1.)
            s += " Rotation {:.1f} degrees.".format(phi)

        if is_interior:
            print s
            multimesh.add(mesh)

    multimesh.build()
    return multimesh

if __name__ == "__main__":

    if MPI.size(mpi_comm_world()) > 1:
        info("Sorry, this demo does not (yet) run in parallel.")
        exit(0)

    # Build multimesh
    multimesh = build_multimesh(args.num_parts, args.N_x, args.no_random_rotation)

    # Assemble linear system
    # FIXME: This does not work yet
    #vol = assemble_multimesh(1*dx(domain=multimesh))

    # Alternative volume calculation
    V = MultiMeshFunctionSpace(multimesh, "Lagrange", 1)
    f = MultiMeshFunction(V)
    f.vector()[:] = 1.0
    vol = assemble_multimesh(f*dX)

    vol_dolfin = multimesh.compute_volume()

    print "Computed volume (quadrature): {}.".format(vol_dolfin)
    print "Computed volume (assemble):   {}.".format(vol)
    print "Error (quadrature): {}.".format(abs(1-vol_dolfin))
    print "Error (assemble):   {}.".format(abs(1-vol))
    assert abs(vol-1) < 10e-10
