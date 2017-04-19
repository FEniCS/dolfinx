# Copyright (C) 2017 August Johansson
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
# First added:  2017-04-19
# Last changed: 2017-04-19
#
# This demo program creates random meshes and checks that the
# compression of the volume and interface quadrature rules yields the
# same volume and area

import argparse
import numpy
from dolfin import *

parser = argparse.ArgumentParser()
parser.add_argument('--num_parts', type=int, help='number of meshes', default=4)
parser.add_argument('--N_x', type=int, help='number of mesh divisions (mesh size)', default=1, required=False)
parser.add_argument('--order', type=int, help='quadrature rule order', default=1, required=False)
#parser.add_argument('--seed', type=numpy.uint32, help='numpy.random seed', default=1, required=False)
args = parser.parse_args()


def build_multimesh(compress_volume, compress_interface):

    # Create multimesh
    multimesh = MultiMesh()
    multimesh.parameters["compress_volume_quadrature"] = compress_volume
    multimesh.parameters["compress_interface_quadrature"] = compress_interface

    # Add background mesh
    mesh = UnitSquareMesh(args.N_x, args.N_x)
    multimesh.add(mesh)

    # Set seed
    #numpy.random.seed(multimesh.num_parts())
    #numpy.random.seed(seed)

    # Add num_parts-1 random sized and rotated rectangular meshes
    while (multimesh.num_parts() < args.num_parts):

        x0, x1 = numpy.sort(numpy.random.rand(2))
        y0, y1 = numpy.sort(numpy.random.rand(2))
        if abs(x1 - x0) < DOLFIN_EPS:
            x1 += DOLFIN_EPS
        if abs(y1 - y0) < DOLFIN_EPS:
            y1 += DOLFIN_EPS

        N_x_part = int(max(abs(x1-x0)*args.N_x, 1))
        N_y_part = int(max(abs(y1-y0)*args.N_x, 1))
        mesh = RectangleMesh(Point(x0, y0), Point(x1, y1),
                             N_x_part, N_y_part)

        # Rotate
        phi = numpy.random.rand()*180
        mesh.rotate(phi)
        coords = mesh.coordinates()
        is_interior = not numpy.any(coords < 0) and not numpy.any(coords > 1.)

        if is_interior:
            multimesh.add(mesh)

    multimesh.build(args.order)
    return multimesh

def volume_area(multimesh):
    volume = multimesh.compute_volume()
    area = multimesh.compute_area()
    return volume, area

if __name__ == "__main__":

    multimesh = build_multimesh(False, False)
    volume, area = volume_area(multimesh)
    print("exact volume {}".format(volume))
    print("exact area {}".format(area))

    # Volume compression
    multimesh_v = build_multimesh(True, False)
    volume_v, area_v = volume_area(multimesh)
    print("volume compressed volume {}".format(volume_v))
    print("volume compressed area {}".format(area_v))

    # Interface compression
    multimesh_i = build_multimesh(False, True)
    volume_i, area_i = volume_area(multimesh_i)
    print("interface compressed volume {}".format(volume_i))
    print("interface compressed area {}".format(area_i))

    # Check errors
    errors = False
    if (abs(volume - volume_v) > 1e-10):
        print(" Error while compressing volume")
        errors = True

    if (abs(area - area_i) > 1e-10):
        print(" Error while compressing area")
        errors = True

    # Errors that should not happen
    if (abs(volume - volume_i) > DOLFIN_EPS):
        print(" Strange behavior: volume changed when compressing interface")
        errors = True

    if (abs(area - area_v) > DOLFIN_EPS):
        print(" Strange behavior: area changed when compressing volume")
        errors = True

    #assert(not errors)
