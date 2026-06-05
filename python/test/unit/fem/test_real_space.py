from mpi4py import MPI

import numpy as np
import pytest

import basix.ufl
import dolfinx
import ufl


@pytest.mark.parametrize("L", [0.1, 0.2, 0.3])
@pytest.mark.parametrize("H", [1.3, 0.8, 0.2])
@pytest.mark.parametrize(
    "cell_type", [dolfinx.mesh.CellType.triangle, dolfinx.mesh.CellType.quadrilateral]
)
@pytest.mark.parametrize("dtype", [np.float64, np.float32])
def test_real_function_space_mass(L, H, cell_type, dtype):
    """Test that real space mass matrix is the same as assembling the volume."""
    mesh = dolfinx.mesh.create_rectangle(
        MPI.COMM_WORLD, [[0.0, 0.0], [L, H]], [7, 9], cell_type, dtype=dtype
    )

    el = basix.ufl.real_element(mesh.basix_cell(), dtype=dtype)
    V = dolfinx.fem.functionspace(mesh, el)
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    a = ufl.inner(u, v) * ufl.dx

    A = dolfinx.fem.assemble_matrix(dolfinx.fem.form(a, dtype=dtype), bcs=[])
    A.scatter_reverse()
    tol = 100 * np.finfo(dtype).eps
    cell_map = mesh.topology.index_map(mesh.topology.dim)
    if cell_map.size_local + cell_map.num_ghosts > 0:
        assert len(A.data) == 1
        if cell_map.local_range[0] == 0:
            assert np.isclose(A.data[0], L * H, atol=tol)
    else:
        assert len(A.data) == 0
        assert len(V.dofmap.list.flatten()) == 0
        assert V.dofmap.index_map.size_local == 0
        assert V.dofmap.index_map.num_ghosts == 1


@pytest.mark.parametrize("dtype", [np.float64, np.float32])
@pytest.mark.parametrize(
    "cell_type", [dolfinx.mesh.CellType.tetrahedron, dolfinx.mesh.CellType.hexahedron]
)
def test_real_function_space_vector(cell_type, dtype):
    """Test assembling with real space test function is equal to assembling with a constant."""
    mesh = dolfinx.mesh.create_unit_cube(MPI.COMM_WORLD, 2, 3, 5, cell_type, dtype=dtype)

    V = dolfinx.fem.functionspace(mesh, ("Lagrange", 3))
    v = ufl.TrialFunction(V)

    el = basix.ufl.real_element(mesh.basix_cell(), dtype=dtype)
    R = dolfinx.fem.functionspace(mesh, el)
    u = ufl.TestFunction(R)
    a_R = ufl.inner(u, v) * ufl.dx
    form_rhs = dolfinx.fem.form(a_R, dtype=dtype)

    A_R = dolfinx.fem.assemble_matrix(form_rhs, bcs=[])
    A_R.scatter_reverse()

    L = ufl.inner(ufl.constantvalue.IntValue(1), v) * ufl.dx
    form_lhs = dolfinx.fem.form(L, dtype=dtype)
    b = dolfinx.fem.assemble_vector(form_lhs)
    b.scatter_reverse(dolfinx.la.InsertMode.add)
    b.scatter_forward()

    row_map = A_R.index_map(0)
    num_local_rows = row_map.size_local
    num_dofs = V.dofmap.index_map.size_local * V.dofmap.index_map_bs
    tol = 100 * np.finfo(dtype).eps
    if MPI.COMM_WORLD.rank == 0:
        assert num_local_rows == 1
        num_dofs_global = V.dofmap.index_map.size_global * V.dofmap.index_map_bs
        assert A_R.indptr[1] - A_R.indptr[0] == num_dofs_global
        np.testing.assert_allclose(A_R.indices, np.arange(num_dofs_global))
        np.testing.assert_allclose(b.array[:num_dofs], A_R.data[:num_dofs], atol=tol)
    else:
        assert num_local_rows == 0


@pytest.mark.parametrize(
    "ftype, stype",
    [
        pytest.param(np.float32, np.complex64, marks=pytest.mark.xfail_win32_complex),
        pytest.param(np.float64, np.complex128, marks=pytest.mark.xfail_win32_complex),
    ],
)
def test_complex_real_space(ftype, stype):
    mesh = dolfinx.mesh.create_unit_interval(MPI.COMM_WORLD, 13, dtype=ftype)

    val = (2 + 3j, -4 + 5j)
    value_shape = (2,)
    el = basix.ufl.real_element(mesh.basix_cell(), value_shape=value_shape, dtype=ftype)
    R = dolfinx.fem.functionspace(mesh, el)
    u = dolfinx.fem.Function(R, dtype=stype)
    u.x.array[0] = val[0]
    u.x.array[1] = val[1]
    u.x.scatter_forward()

    V = dolfinx.fem.functionspace(mesh, ("Lagrange", 1, value_shape))
    v = ufl.TestFunction(V)
    L = ufl.inner(u, v) * ufl.dx

    b = dolfinx.fem.assemble_vector(dolfinx.fem.form(L, dtype=stype))
    b.scatter_reverse(dolfinx.la.InsertMode.add)
    b.scatter_forward()
    const = dolfinx.fem.Constant(mesh, stype(val))
    L_const = ufl.inner(const, v) * ufl.dx
    b_const = dolfinx.fem.assemble_vector(dolfinx.fem.form(L_const, dtype=stype))
    b_const.scatter_reverse(dolfinx.la.InsertMode.add)
    b_const.scatter_forward()

    tol = 100 * np.finfo(stype).eps
    np.testing.assert_allclose(b.array, b_const.array, atol=tol)
