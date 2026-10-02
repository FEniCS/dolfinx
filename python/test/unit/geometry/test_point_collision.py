from mpi4py import MPI

import numpy as np
import pytest

import basix
import dolfinx
import ufl


@pytest.mark.skip_in_parallel  # Skip in parallel as there is only one cell
@pytest.mark.parametrize("find_closest_cell", [True, False])
def test_determine_point_ownership(find_closest_cell):
    comm = MPI.COMM_WORLD

    # Define mesh consiting of a single tetrahedron
    rank = comm.rank
    if rank == 0:
        points = np.array(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64
        )

        cells = np.array([[0, 1, 2, 3]], dtype=np.int64)
    else:
        points = np.empty((0, 3), dtype=np.float64)
        cells = np.empty((0, 4), dtype=np.int64)
    coord_element = basix.ufl.element("Lagrange", "tetrahedron", 1, shape=(3,))
    ufl_mesh = ufl.Mesh(coord_element)
    mesh = dolfinx.mesh.create_mesh(comm, cells=cells, e=ufl_mesh, x=points)

    # Define points, for which collisions with the mesh will be detected
    if comm.rank == 0:
        points_ext = np.array(
            [
                [0.25, 0.25, 0.25],  # Point inside the mesh
                [0.5, 0.5, 0.5],  # Point outside the mesh, but inside the bounding box
                [1.5, 0.5, 0.5],  # Point outside the bounding box
            ],
            dtype=np.float64,
        )
    else:
        points_ext = np.empty((0, 3), dtype=np.float64)

    # Check, which rank contains the point, if any does
    point_owner_ship_data = dolfinx.geometry.determine_point_ownership(
        mesh, points_ext, padding=0.0, find_closest_cell=find_closest_cell
    )

    dest_cells = point_owner_ship_data.dest_cells
    src_owner = point_owner_ship_data.src_owner

    num_cells = mesh.geometry.dofmaps[0].shape[0]
    if num_cells > 0:
        if find_closest_cell:
            assert len(dest_cells) == 2
        else:
            assert len(dest_cells) == 1

    if rank == 0:
        if find_closest_cell:
            assert len(np.flatnonzero(src_owner != -1)) == 2
            assert src_owner[2] == -1
        else:
            assert len(np.flatnonzero(src_owner != -1)) == 1
            assert src_owner[2] == -1
