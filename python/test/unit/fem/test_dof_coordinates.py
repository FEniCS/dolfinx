import numpy as np
import pytest

from dolfinx.fem import Function, FunctionSpace
from dolfinx.mesh import (create_unit_cube, create_unit_square,
                          create_rectangle, GhostMode,
                          locate_entities, create_submesh)


from mpi4py import MPI


@pytest.mark.parametrize("degree", range(1, 5))
def test_dof_coords_2d(degree):
    mesh = create_unit_square(MPI.COMM_WORLD, 10, 10)
    V = FunctionSpace(mesh, ("Lagrange", degree))
    u = Function(V)
    u.interpolate(lambda x: x[0])
    u.x.scatter_forward()
    x = V.tabulate_dof_coordinates()
    val = u.vector.array
    for i in range(len(val)):
        assert np.isclose(x[i, 0], val[i], rtol=1e-3)


@pytest.mark.parametrize("degree", range(1, 5))
def test_dof_coords_3d(degree):
    mesh = create_unit_cube(MPI.COMM_WORLD, 10, 10, 10)
    V = FunctionSpace(mesh, ("Lagrange", degree))
    u = Function(V)
    u.interpolate(lambda x: x[0])
    u.x.scatter_forward()
    x = V.tabulate_dof_coordinates()
    val = u.vector.array
    for i in range(len(val)):
        assert np.isclose(x[i, 0], val[i], rtol=1e-3)


@pytest.mark.parametrize("ghost_mode", [GhostMode.none,
                                        GhostMode.shared_facet])
def test_dof_not_in_cell(ghost_mode):
    """Test tabulating dof coordinates where some dofs don't
    belong to any cells on some processes"""

    # Currently, it doesn't seem to be possible to directly
    # create a mesh where some some dofs don't belong to a
    # cell on some processes, so create one indirectly with
    # `create_submesh`
    n = 1
    mesh = create_rectangle(
        MPI.COMM_WORLD, ((0.0, 0.0), (2.0, 1.0)), (2 * n, n),
        ghost_mode=ghost_mode)
    edim = mesh.topology.dim
    entities = locate_entities(mesh, edim, lambda x: x[0] <= 1.0)
    submesh = create_submesh(mesh, edim, entities)[0]

    # Create a function space over the submesh, and tabulate the
    # dof coordinates
    V = FunctionSpace(submesh, ("Lagrange", 1))
    coords = V.tabulate_dof_coordinates()

    # Round any small values to zero
    coords[np.isclose(coords, 0.0)] = 0

    # Check that there is at most one dof on each process
    # positioned at (0, 0, 0).
    assert(np.sum(~coords.any(1)) <= 1)
