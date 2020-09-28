import dolfinx
import dolfinx.cpp.mesh as cmesh
import dolfinx.fem
import dolfinx.io
import dolfinx.mesh
import numpy as np
import ufl
from mpi4py import MPI


def create_boundary_mesh(mesh, comm):
    """
    Create a mesh consisting of all exterior facets of a mesh
    Input:
      mesh - The mesh
    Output:
      bmesh - The boundary mesh
      bmesh_to_geometry - Map from cells of the boundary mesh
                          to the geometry of the original mesh
    """
    ext_facets = cmesh.exterior_facet_indices(mesh)
    boundary_geometry = cmesh.entities_to_geometry(
        mesh, mesh.topology.dim - 1, ext_facets, True)
    facet_type = dolfinx.cpp.mesh.to_string(cmesh.cell_entity_type(
        mesh.topology.cell_type, mesh.topology.dim - 1))
    facet_cell = ufl.Cell(facet_type,
                          geometric_dimension=mesh.geometry.dim)
    degree = mesh.ufl_domain().ufl_coordinate_element().degree()
    ufl_domain = ufl.Mesh(ufl.VectorElement("Lagrange", facet_cell, degree))
    bmesh = dolfinx.mesh.create_mesh(
        comm, boundary_geometry, mesh.geometry.x, ufl_domain)
    return bmesh, boundary_geometry


def test_b_mesh_mapping():
    mesh = dolfinx.UnitCubeMesh(MPI.COMM_WORLD, 2, 2, 2)

    b_mesh, bndry_to_mesh = create_boundary_mesh(mesh, MPI.COMM_SELF)

    # Compute map from boundary mesh topology to boundary mesh geometry
    b_mesh.topology.create_connectivity(
        b_mesh.topology.dim, b_mesh.topology.dim)
    b_imap = b_mesh.topology.index_map(b_mesh.topology.dim)
    tdim_entities = np.arange(b_imap.size_local * b_imap.block_size,
                              dtype=np.int32)
    boundary_geometry = cmesh.entities_to_geometry(
        b_mesh, b_mesh.topology.dim, tdim_entities, False)

    # Compare geometry maps
    for i in range(boundary_geometry.shape[0]):
        assert(
            np.allclose(b_mesh.geometry.x[boundary_geometry[i]],
                        mesh.geometry.x[bndry_to_mesh[i]]))

    # Check that boundary mesh integrated has the correct area
    b_volume = mesh.mpi_comm().allreduce(dolfinx.fem.assemble_scalar(
        dolfinx.Constant(b_mesh, 1) * ufl.dx), op=MPI.SUM)
    mesh_surface = mesh.mpi_comm().allreduce(dolfinx.fem.assemble_scalar(
        dolfinx.Constant(mesh, 1) * ufl.ds), op=MPI.SUM)
    assert(np.isclose(b_volume, mesh_surface))
