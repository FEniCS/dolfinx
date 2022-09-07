# Copyright (C) 2022 Michele Castriotta, Igor Baratta, Jørgen S. Dokken

import sys

try:
    import gmsh
except ModuleNotFoundError:
    print("This demo requires gmsh to be installed")
    sys.exit(0)
from numpy import pi

from mpi4py import MPI


def generate_mesh_wire(
    radius_wire: float, radius_dom: float, in_wire_size: float,
    on_wire_size: float, bkg_size: float, boundary_size: float,
        au_tag: int, bkg_tag: int, boundary_tag: int):

    gmsh.initialize(sys.argv)
    if MPI.COMM_WORLD.rank == 0:

        gmsh.model.add("nanowire")

        # A dummy boundary is added for setting a finer mesh
        gmsh.model.occ.addCircle(0.0, 0.0, 0.0, radius_wire * 0.8,
                                 angle1=0.0, angle2=2 * pi, tag=1)
        gmsh.model.occ.addCircle(0.0, 0.0, 0.0, radius_wire,
                                 angle1=0, angle2=2 * pi, tag=2)

        # A dummy boundary is added for setting a finer mesh
        gmsh.model.occ.addCircle(0.0, 0.0, 0.0, radius_dom * 0.9,
                                 angle1=0.0, angle2=2 * pi, tag=3)
        gmsh.model.occ.addCircle(
            0.0, 0.0, 0.0, radius_dom, angle1=0.0, angle2=2 * pi, tag=4)

        gmsh.model.occ.addCurveLoop([1], tag=1)
        gmsh.model.occ.addPlaneSurface([1], tag=1)

        gmsh.model.occ.addCurveLoop([2], tag=2)
        gmsh.model.occ.addCurveLoop([1], tag=3)
        gmsh.model.occ.addPlaneSurface([2, 3], tag=2)

        gmsh.model.occ.addCurveLoop([3], tag=4)
        gmsh.model.occ.addCurveLoop([2], tag=5)
        gmsh.model.occ.addPlaneSurface([4, 5], tag=3)

        gmsh.model.occ.addCurveLoop([4], tag=6)
        gmsh.model.occ.addCurveLoop([3], tag=7)
        gmsh.model.occ.addPlaneSurface([6, 7], tag=4)

        gmsh.model.occ.synchronize()

        gmsh.model.addPhysicalGroup(2, [1, 2], tag=au_tag)
        gmsh.model.addPhysicalGroup(2, [3, 4], tag=bkg_tag)

        gmsh.model.addPhysicalGroup(1, [4], tag=boundary_tag)

        gmsh.model.mesh.setSize([(0, 1)], size=in_wire_size)
        gmsh.model.mesh.setSize([(0, 2)], size=on_wire_size)
        gmsh.model.mesh.setSize([(0, 3)], size=bkg_size)
        gmsh.model.mesh.setSize([(0, 4)], size=boundary_size)

        gmsh.model.mesh.generate(2)

        return gmsh.model
