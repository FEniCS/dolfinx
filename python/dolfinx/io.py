# Copyright (C) 2017-2022 Chris N. Richardson, Garth N. Wells, Michal Habera
# and Jørgen S. Dokken
#
# This file is part of DOLFINx (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
"""IO module for input data, post-processing file outout and
checkpointing"""

import typing

import numpy as np
import numpy.typing as npt

import ufl
from dolfinx import cpp as _cpp
from dolfinx.fem import Function
from dolfinx.mesh import CellType, GhostMode, Mesh

from mpi4py import MPI as _MPI

__all__ = ["VTKFile", "XDMFFile", "cell_perm_gmsh", "distribute_entity_data"]


def _extract_cpp_functions(functions: typing.Union[typing.List[Function], Function]):
    """Extract C++ object for a single function or a list of functions"""
    if isinstance(functions, (list, tuple)):
        return [getattr(u, "_cpp_object", u) for u in functions]
    else:
        return [getattr(functions, "_cpp_object", functions)]


if _cpp.common.has_adios2:
    # FidesWriter and VTXWriter require ADIOS2

    __all__ = __all__ + ["FidesWriter", "VTXWriter"]

    class VTXWriter(_cpp.io.VTXWriter):
        """Interface to VTK files for ADIOS2

        VTX supports arbitrary order Lagrange finite elements for the
        geometry description and arbitrary order (discontinuous)
        Lagrange finite elements for Functions.

        The files can be displayed by Paraview. The storage backend uses
        ADIOS2.

        """

        def __init__(self, comm: _MPI.Comm, filename: str, output: typing.Union[Mesh, typing.List[Function], Function]):
            """Initialize a writer for outputting data in the VTX format.

            Args:
                comm: The MPI communicator
                filename: The output filename
                output: The data to output. Either a mesh, a single
                    (discontinuous) Lagrange Function or list of
                    (discontinuous Lagrange Functions.

            Note:
                All Functions for output must share the same mesh and
                have the same element type.

            """
            try:
                # Input is a mesh
                super().__init__(comm, filename, output)
            except (NotImplementedError, TypeError):
                # Input is a single function or a list of functions
                super().__init__(comm, filename, _extract_cpp_functions(output))

        def __enter__(self):
            return self

        def __exit__(self, exception_type, exception_value, traceback):
            self.close()

    class FidesWriter(_cpp.io.FidesWriter):
        """Interface to Fides file formt.

        Fides supports first order Lagrange finite elements for the
        geometry descriptionand first order Lagrange finite elements for
        functions. All functions has to be of the same element family
        and same order.

        The files can be displayed by Paraview. The storage backend uses
        ADIOS2.

        """

        def __init__(self, comm: _MPI.Comm, filename: str, output: typing.Union[Mesh, typing.List[Function], Function]):
            """Initialize a writer for outputting a mesh, a single Lagrange
            function or list of Lagrange functions sharing the same
            element family and degree

            Args:
                comm: The MPI communicator
                filename: The output filename
                output: The data to output. Either a mesh, a single
                    first order Lagrange function or list of first order
                    Lagrange functions.

            """
            try:
                super().__init__(comm, filename, output)
            except (NotImplementedError, TypeError):
                super().__init__(comm, filename, _extract_cpp_functions(output))

        def __enter__(self):
            return self

        def __exit__(self, exception_type, exception_value, traceback):
            self.close()


class VTKFile(_cpp.io.VTKFile):
    """Interface to VTK files

    VTK supports arbitrary order Lagrange finite elements for the
    geometry description. XDMF is the preferred format for geometry
    order <= 2.

    """

    def __enter__(self):
        return self

    def __exit__(self, exception_type, exception_value, traceback):
        self.close()

    def write_mesh(self, mesh: Mesh, t: float = 0.0) -> None:
        """Write mesh to file for a given time (default 0.0)"""
        self.write(mesh, t)

    def write_function(self, u: typing.Union[typing.List[Function], Function], t: float = 0.0) -> None:
        """Write a single function or a list of functions to file for a given time (default 0.0)"""
        super().write(_extract_cpp_functions(u), t)


class XDMFFile(_cpp.io.XDMFFile):
    def __enter__(self):
        return self

    def __exit__(self, exception_type, exception_value, traceback):
        self.close()

    def write_mesh(self, mesh: Mesh) -> None:
        """Write mesh to file for a given time (default 0.0)"""
        super().write_mesh(mesh)

    def write_function(self, u, t: float = 0.0, mesh_xpath="/Xdmf/Domain/Grid[@GridType='Uniform'][1]"):
        super().write_function(getattr(u, "_cpp_object", u), t, mesh_xpath)

    def read_mesh(self, ghost_mode=GhostMode.shared_facet, name="mesh", xpath="/Xdmf/Domain") -> Mesh:
        """Read mesh data from file"""
        cell_shape, cell_degree = super().read_cell_type(name, xpath)
        cells = super().read_topology_data(name, xpath)
        x = super().read_geometry_data(name, xpath)

        # Construct the geometry map
        cell = ufl.Cell(cell_shape.name, geometric_dimension=x.shape[1])

        # Build the mesh
        cmap = _cpp.fem.CoordinateElement(cell_shape, cell_degree)
        mesh = _cpp.mesh.create_mesh(self.comm(), _cpp.graph.AdjacencyList_int64(cells),
                                     cmap, x, ghost_mode, _cpp.mesh.create_cell_partitioner())
        mesh.name = name

        domain = ufl.Mesh(ufl.VectorElement("Lagrange", cell, cell_degree))
        return Mesh.from_cpp(mesh, domain)

    def read_meshtags(self, mesh, name, xpath="/Xdmf/Domain"):
        return super().read_meshtags(mesh, name, xpath)


def distribute_entity_data(mesh: Mesh, entity_dim: int, entities: npt.NDArray[np.int64],
                           values: npt.NDArray[np.int32]) -> typing.Tuple[npt.NDArray[np.int64], npt.NDArray[np.int32]]:
    return _cpp.io.distribute_entity_data(mesh, entity_dim, entities, values)


def cell_perm_gmsh(cell_type: CellType, dim: int) -> typing.List[int]:
    return _cpp.io.perm_gmsh(cell_type, dim)


def extract_gmsh_topology_and_markers(gmsh_model, model_name=None):
    """Extract all entities tagged with a physical marker in the gmsh
    model, and collect the data per cell type. Returns a nested
    dictionary where the first key is the gmsh MSH element type integer.
    Each element type present in the model contains the cell topology of
    the elements and corresponding markers.

    """
    if model_name is not None:
        gmsh_model.setCurrent(model_name)

    # Get the physical groups from gmsh on the form [(dim1, tag1),(dim1,
    # tag2), (dim2, tag3),...]
    phys_grps = gmsh_model.getPhysicalGroups()
    topologies = {}
    for dim, tag in phys_grps:
        # Get the entities for a given dimension:
        # dim=0->Points, dim=1->Lines, dim=2->Triangles/Quadrilaterals,
        # etc.
        entities = gmsh_model.getEntitiesForPhysicalGroup(dim, tag)

        for entity in entities:
            # Get data about the elements on a given entity:
            # NOTE: Assumes that each entity only have one cell-type
            element_data = gmsh_model.mesh.getElements(dim, tag=entity)
            element_types, element_tags, node_tags = element_data
            assert len(element_types) == 1

            # The MSH type of the cells on the element
            element_type = element_types[0]
            num_el = len(element_tags[0])

            # Determine number of local nodes per element to create the
            # topology of the elements
            properties = gmsh_model.mesh.getElementProperties(element_type)
            name, dim, order, num_nodes, local_coords, _ = properties

            # 2D array of shape (num_elements,num_nodes_per_element)
            # containing the topology of the elements on this entity
            # NOTE: GMSH indexing starts with 1 and not zero
            element_topology = node_tags[0].reshape(-1, num_nodes) - 1

            # Gather data for each element type and the
            # corresponding physical markers
            if element_type in topologies.keys():
                topologies[element_type]["topology"] = np.concatenate(
                    (topologies[element_type]["topology"], element_topology), axis=0)
                topologies[element_type]["cell_data"] = np.concatenate(
                    (topologies[element_type]["cell_data"], np.full(num_el, tag)), axis=0)
            else:
                topologies[element_type] = {"topology": element_topology, "cell_data": np.full(num_el, tag)}

    return topologies


def extract_gmsh_geometry(gmsh_model, model_name=None):
    """For a given gmsh model, extract the mesh geometry as a numpy
    (N, 3) array where the i-th row corresponds to the i-th node in the
    mesh.

    """
    if model_name is not None:
        gmsh_model.setCurrent(model_name)

    # Get the unique tag and coordinates for nodes
    # in mesh
    indices, points, _ = gmsh_model.mesh.getNodes()
    points = points.reshape(-1, 3)

    # GMSH indices starts at 1
    indices -= 1

    # Sort nodes in geometry according to the unique index
    perm_sort = np.argsort(indices)
    assert np.all(indices[perm_sort] == np.arange(len(indices)))
    return points[perm_sort]


# Map from Gmsh int to DOLFINx cell type and degree
# http://gmsh.info//doc/texinfo/gmsh.html#MSH-file-format
_gmsh_to_cells = {1: ("interval", 1), 2: ("triangle", 1),
                  3: ("quadrilateral", 1), 4: ("tetrahedron", 1),
                  5: ("hexahedron", 1), 8: ("interval", 2),
                  9: ("triangle", 2), 10: ("quadrilateral", 2),
                  11: ("tetrahedron", 2), 12: ("hexahedron", 2),
                  15: ("point", 0), 21: ("triangle", 3),
                  26: ("interval", 3), 29: ("tetrahedron", 3),
                  36: ("quadrilateral", 3)}


def ufl_mesh_from_gmsh(gmsh_cell: int, gdim: int) -> ufl.Mesh:
    """Create a UFL mesh from a Gmsh cell identifier and the geometric dimension.
    See: http://gmsh.info//doc/texinfo/gmsh.html#MSH-file-format.

    """
    shape, degree = _gmsh_to_cells[gmsh_cell]
    cell = ufl.Cell(shape, geometric_dimension=gdim)
    scalar_element = ufl.FiniteElement("Lagrange", cell, degree, variant="equispaced")
    return ufl.Mesh(ufl.VectorElement(scalar_element))
