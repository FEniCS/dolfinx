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
# This demo program ...

import argparse
from dolfin import *

parser = argparse.ArgumentParser()
parser.add_argument('--max_num_meshes', type=int, help='maximum number of meshes', default=10)
args = parser.parse_args()

print("Maximum number of meshes: {}.".format(args.max_num_meshes))


def test_volume_2d_rot(num_meshes):

    # Create multimesh
    multimesh = MultiMesh()

    # Add background mesh
    mesh = RectangleMesh(Point(-5, -5), Point(5, 5), 1, 1)
    multimesh.add(mesh)

    # Add rotated meshes
    for i in range(num_meshes):

        mesh = RectangleMesh(Point(-1, -1), Point(1, 1), 1, 1)

        angle = 2 * DOLFIN_PI * i / num_meshes
        mesh.rotate(180.0 * angle / DOLFIN_PI)
        mesh.translate(Point(cos(angle), sin(angle)))
        multimesh.add(mesh)

    multimesh.build()
    return multimesh

if __name__ == "__main__":

    if MPI.size(mpi_comm_world()) > 1:
        info("Sorry, this demo does not (yet) run in parallel.")
        exit(0)

    for num_meshes in range(args.max_num_meshes):

        print "Testing with {} meshes.".format(num_meshes)

        # Build multimesh
        multimesh = test_volume_2d_rot(num_meshes)

        # Assemble linear system
        # FIXME: This does not work yet
        #vol = assemble_multimesh(1*dx(domain=multimesh))

        # Alternative volume calculation
        V = MultiMeshFunctionSpace(multimesh, "Lagrange", 1)
        f = MultiMeshFunction(V)
        f.vector()[:] = 1.0
        vol = assemble_multimesh(f*dX)
        
        print "Computed volume: {}.".format(vol)
        print "Error: {}.".format(abs(100-vol))
        assert abs(vol-100) < 10e-10
