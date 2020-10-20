import dolfinx
import dolfinx.io
from mpi4py import MPI
import numpy as np
import ufl

mesh = dolfinx.UnitSquareMesh(MPI.COMM_WORLD, 2, 2)


def boundary(x):
    return np.isclose(x[0], np.finfo(float).eps)


fdim = mesh.topology.dim - 1
facets = dolfinx.mesh.locate_entities_boundary(mesh, fdim,
                                               boundary)
values = np.full(len(facets), 1, dtype=np.int32)
ft = dolfinx.MeshTags(mesh, fdim, facets, values)
ft.name = "Facet data"

num_cells = mesh.topology.index_map(mesh.topology.dim).size_local
cell_values = np.full(num_cells, 1, dtype=np.int32)
ct = dolfinx.MeshTags(mesh, mesh.topology.dim,
                      np.arange(num_cells), cell_values)
ct.name = "Cell data"

with dolfinx.io.XDMFFile(MPI.COMM_WORLD,
                         "mesh.xdmf", "w") as xdmf:
    geometry_xpath = "/Xdmf/Domain/Grid[@Name='geometry']/Geometry"
    xdmf.write_geometry(mesh.geometry)
    xdmf.write_meshtags(ct, geometry_xpath=geometry_xpath)
    xdmf.write_meshtags(ft, geometry_xpath=geometry_xpath)


with dolfinx.io.XDMFFile(MPI.COMM_WORLD, "mesh.xdmf", "r") as xdmf:
    geometry = xdmf.read_geometry_data(name="geometry")
    topology = xdmf.read_topology_data(name="Cell data")
    ufl_mesh = ufl.Mesh(ufl.VectorElement("Lagrange", ufl.Cell(
        "triangle", geometric_dimension=geometry.shape[1]), 1))
    cmap = dolfinx.fem.create_coordinate_map(ufl_mesh)
    mesh_in = dolfinx.cpp.mesh.create_mesh(
        MPI.COMM_WORLD, dolfinx.cpp.graph.AdjacencyList_int64(
            topology), cmap, geometry, dolfinx.cpp.mesh.GhostMode.shared_facet)
    mesh_in._ufl_domain = ufl_mesh
    ufl_mesh._ufl_cargo = mesh_in
    mesh_in.topology.create_connectivity(
        mesh_in.topology.dim-1, mesh_in.topology.dim)
    ct_in = xdmf.read_meshtags(mesh_in, name="Cell data")
    ft_in = xdmf.read_meshtags(mesh_in, name="Facet data")
print(MPI.COMM_WORLD.rank, ft.values, ft_in.values,
      "\n", ft.indices, ft_in.indices)
print(MPI.COMM_WORLD.rank,
      mesh.topology.index_map(mesh.topology.dim).size_local,
      mesh_in.topology.index_map(mesh_in.topology.dim).size_local)
