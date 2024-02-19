# # Mesh generation for the wire and the PML with Gmsh
#
# Copyright (C) 2022 Michele Castriotta, Igor Baratta, Jørgen S. Dokken
#
# This file defines the `generate_mesh_wire` function, which is used to
# generate the mesh used for the PML demo. The mesh is made up by a
# central circle (the wire), and an external layer (the PML) divided in
# 4 rectangles and 4 squares at the corners. The `generate_mesh_wire`
# function takes as input:

# - `radius_wire`: the radius of the wire
# - `radius_scatt`: the radius of the circle where scattering efficiency
#   is calculated
# - `l_dom`: length of real domain
# - `l_pml`: length of PML layer
# - `in_wire_size`: the mesh size at a distance `0.8 * radius_wire` from
#   the origin
# - `on_wire_size`: the mesh size on the wire boundary
# - `scatt_size`: the mesh size on the circle where scattering
#   efficiency is calculated
# - `pml_size`: the mesh size on the outer boundary of the PML
# - `au_tag`: the tag of the physical group representing the wire
# - `bkg_tag`: the tag of the physical group representing the background
# - `scatt_tag`: the tag of the physical group representing the boundary
#   where scattering efficiency is calculated
# - `pml_tag`: the tag of the physical group representing the PML
#   (together with pml_tag+1 and pml_tag+2)
#
#

import sys
from functools import reduce

import numpy.typing

try:
    import gmsh
except ModuleNotFoundError:
    print("This demo requires gmsh to be installed")
    sys.exit(0)
from numpy import intersect1d, pi


def generate_mesh_wire(
    radius_wire: float,
    radius_scatt: float,
    l_dom: float,
    l_pml: float,
    in_wire_size: float,
    on_wire_size: float,
    scatt_size: float,
    pml_size: float,
    au_tag: int,
    bkg_tag: int,
    scatt_tag: int,
    pml_tag: int,
):
    gmsh.model.add("nanowire")
    dim = 2
    # A dummy circle for setting a finer mesh
    c1 = gmsh.model.occ.addCircle(0.0, 0.0, 0.0, radius_wire * 0.8, angle1=0.0, angle2=2 * pi)
    gmsh.model.occ.addCurveLoop([c1], tag=c1)
    gmsh.model.occ.addPlaneSurface([c1], tag=c1)

    c2 = gmsh.model.occ.addCircle(0.0, 0.0, 0.0, radius_wire, angle1=0, angle2=2 * pi)
    gmsh.model.occ.addCurveLoop([c2], tag=c2)
    gmsh.model.occ.addPlaneSurface([c2], tag=c2)
    wire, _ = gmsh.model.occ.fragment([(dim, c2)], [(dim, c1)])

    # A dummy circle for the calculation of the scattering efficiency
    c3 = gmsh.model.occ.addCircle(0.0, 0.0, 0.0, radius_scatt, angle1=0, angle2=2 * pi)
    gmsh.model.occ.addCurveLoop([c3], tag=c3)
    gmsh.model.occ.addPlaneSurface([c3], tag=c3)

    r0 = gmsh.model.occ.addRectangle(-l_dom / 2, -l_dom / 2, 0, l_dom, l_dom)
    inclusive_rectangle, _ = gmsh.model.occ.fragment([(dim, r0)], [(dim, c3)])

    delta_pml = (l_pml - l_dom) / 2

    separate_rectangle, _ = gmsh.model.occ.cut(inclusive_rectangle, wire, removeTool=False)
    _, physical_domain = gmsh.model.occ.fragment(separate_rectangle, wire)

    bkg_tags = [tag[0] for tag in physical_domain[: len(separate_rectangle)]]

    wire_tags = [
        tag[0]
        for tag in physical_domain[len(separate_rectangle) : len(inclusive_rectangle) + len(wire)]
    ]

    # Corner PMLS
    pml1 = gmsh.model.occ.addRectangle(-l_pml / 2, l_dom / 2, 0, delta_pml, delta_pml)
    pml2 = gmsh.model.occ.addRectangle(-l_pml / 2, -l_pml / 2, 0, delta_pml, delta_pml)
    pml3 = gmsh.model.occ.addRectangle(l_dom / 2, l_dom / 2, 0, delta_pml, delta_pml)
    pml4 = gmsh.model.occ.addRectangle(l_dom / 2, -l_pml / 2, 0, delta_pml, delta_pml)
    corner_pmls = [(dim, pml1), (dim, pml2), (dim, pml3), (dim, pml4)]

    # X pmls
    pml5 = gmsh.model.occ.addRectangle(-l_pml / 2, -l_dom / 2, 0, delta_pml, l_dom)
    pml6 = gmsh.model.occ.addRectangle(l_dom / 2, -l_dom / 2, 0, delta_pml, l_dom)
    x_pmls = [(dim, pml5), (dim, pml6)]

    # Y pmls
    pml7 = gmsh.model.occ.addRectangle(-l_dom / 2, l_dom / 2, 0, l_dom, delta_pml)
    pml8 = gmsh.model.occ.addRectangle(-l_dom / 2, -l_pml / 2, 0, l_dom, delta_pml)
    y_pmls = [(dim, pml7), (dim, pml8)]
    _, surface_map = gmsh.model.occ.fragment(bkg_tags + wire_tags, corner_pmls + x_pmls + y_pmls)

    gmsh.model.occ.synchronize()

    bkg_group = [tag[0][1] for tag in surface_map[: len(bkg_tags)]]
    gmsh.model.addPhysicalGroup(dim, bkg_group, tag=bkg_tag)
    wire_group = [tag[0][1] for tag in surface_map[len(bkg_tags) : len(bkg_tags + wire_tags)]]

    gmsh.model.addPhysicalGroup(dim, wire_group, tag=au_tag)

    corner_group = [
        tag[0][1]
        for tag in surface_map[len(bkg_tags + wire_tags) : len(bkg_tags + wire_tags + corner_pmls)]
    ]
    gmsh.model.addPhysicalGroup(dim, corner_group, tag=pml_tag)

    x_group = [
        tag[0][1]
        for tag in surface_map[
            len(bkg_tags + wire_tags + corner_pmls) : len(
                bkg_tags + wire_tags + corner_pmls + x_pmls
            )
        ]
    ]

    gmsh.model.addPhysicalGroup(dim, x_group, tag=pml_tag + 1)

    y_group = [
        tag[0][1]
        for tag in surface_map[
            len(bkg_tags + wire_tags + corner_pmls + x_pmls) : len(
                bkg_tags + wire_tags + corner_pmls + x_pmls + y_pmls
            )
        ]
    ]

    gmsh.model.addPhysicalGroup(dim, y_group, tag=pml_tag + 2)

    # Marker interior surface in bkg group
    boundaries: list[numpy.typing.NDArray[numpy.int32]] = []
    for tag in bkg_group:
        boundary_pairs = gmsh.model.get_boundary([(dim, tag)], oriented=False)
        boundaries.append(numpy.asarray([pair[1] for pair in boundary_pairs], dtype=numpy.int32))

    interior_boundary = reduce(intersect1d, boundaries)
    gmsh.model.addPhysicalGroup(dim - 1, interior_boundary, tag=scatt_tag)
    gmsh.model.mesh.setSize([(0, 1)], size=in_wire_size)
    gmsh.model.mesh.setSize([(0, 2)], size=on_wire_size)
    gmsh.model.mesh.setSize([(0, 3)], size=scatt_size)
    gmsh.model.mesh.setSize([(0, x) for x in range(4, 40)], size=pml_size)

    gmsh.model.mesh.generate(2)
    return gmsh.model
