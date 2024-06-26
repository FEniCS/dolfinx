from mpi4py import MPI

import numpy as np
import pytest

import basix.ufl
import dolfinx
import ufl


@pytest.mark.parametrize(
    "dtype",
    [
        np.float32,
        np.float64,
        pytest.param(np.complex64, marks=pytest.mark.xfail_win32_complex),
        pytest.param(np.complex128, marks=pytest.mark.xfail_win32_complex),
    ],
)
def test_compiled_form(dtype):
    """
    Compile a form without an associated mesh and assemble a form over a sequence of meshes
    """
    real_type = dtype(0).real.dtype
    c_el = basix.ufl.element("Lagrange", "triangle", 1, shape=(2,), dtype=real_type)
    domain = ufl.Mesh(c_el)
    el = basix.ufl.element("Lagrange", "triangle", 2, dtype=real_type)
    V = ufl.FunctionSpace(domain, el)
    u = ufl.Coefficient(V)
    w = ufl.Coefficient(V)
    c = ufl.Constant(domain)
    e = ufl.Constant(domain)
    J = c * e * u * w * ufl.dx(domain=domain)

    # Compile form using dolfinx.jit.ffcx_jit
    compiled_form = dolfinx.fem.compile_form(
        MPI.COMM_WORLD, J, form_compiler_options={"scalar_type": dtype}
    )

    def create_and_integrate(N, compiled_form):
        mesh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, N, N, dtype=real_type)
        assert mesh.ufl_domain().ufl_coordinate_element() == c_el
        Vh = dolfinx.fem.functionspace(mesh, u.ufl_element())
        uh = dolfinx.fem.Function(Vh, dtype=dtype)
        uh.interpolate(lambda x: x[0])
        wh = dolfinx.fem.Function(Vh, dtype=dtype)
        wh.interpolate(lambda x: x[1])
        eh = dolfinx.fem.Constant(mesh, dtype(3.0))
        ch = dolfinx.fem.Constant(mesh, dtype(2.0))
        form = dolfinx.fem.create_form(compiled_form, [], mesh, {}, {u: uh, w: wh}, {c: ch, e: eh})
        assert np.isclose(mesh.comm.allreduce(dolfinx.fem.assemble_scalar(form), op=MPI.SUM), 1.5)

    # Create various meshes, that all uses this compiled form with a map from ufl
    # to dolfinx functions and constants
    for i in range(1, 4):
        create_and_integrate(i, compiled_form)





@pytest.mark.parametrize(
    "dtype",
    [
        np.float32,
        np.float64,
        pytest.param(np.complex64, marks=pytest.mark.xfail_win32_complex),
        pytest.param(np.complex128, marks=pytest.mark.xfail_win32_complex),
    ],
)
def test_submesh_assembly(dtype):
    """
    Compile a form without an associated mesh and assemble a form over a sequence of meshes
    """
    real_type = dtype(0).real.dtype
    c_el = basix.ufl.element("Lagrange", "triangle", 1, shape=(2,), dtype=real_type)
    domain = ufl.Mesh(c_el)
    el = basix.ufl.element("Lagrange", "triangle", 2, dtype=real_type)
    V = ufl.FunctionSpace(domain, el)
    u = ufl.Coefficient(V)

    f_el = basix.ufl.element("Lagrange", "interval", 1, shape=(2,), dtype=real_type)
    submesh = ufl.Mesh(f_el)
    sub_el = basix.ufl.element("Lagrange", "interval", 3, dtype=real_type)
    V_sub = ufl.FunctionSpace(submesh, sub_el)

    w = ufl.Coefficient(V_sub)

    subdomain_id = 1
    J = u * w * ufl.ds(domain=domain, subdomain_id=subdomain_id)

    # Compile form using dolfinx.jit.ffcx_jit
    compiled_form = dolfinx.fem.compile_form(
        MPI.COMM_WORLD, J, form_compiler_options={"scalar_type": dtype}
    )

    def create_and_integrate(N, compiled_form):
        mesh = dolfinx.mesh.create_rectangle(MPI.COMM_WORLD,[np.array([0, 0]), np.array([2, 2])],
                                             [N, N], dolfinx.mesh.CellType.triangle, dtype=real_type)
        assert mesh.ufl_domain().ufl_coordinate_element() == c_el

        facets = dolfinx.mesh.locate_entities_boundary(
            mesh, mesh.topology.dim - 1, lambda x: np.isclose(x[1], 2))
        submesh, sub_to_parent, _, _ = dolfinx.mesh.create_submesh(mesh, mesh.topology.dim-1, facets)
        imap =  mesh.topology.index_map(mesh.topology.dim-1)
        num_facets = imap.size_local + imap.num_ghosts
        parent_to_sub = np.zeros(num_facets, dtype=np.int32)
        parent_to_sub[sub_to_parent] = np.arange(sub_to_parent.size, dtype=np.int32)

        Vh = dolfinx.fem.functionspace(mesh, u.ufl_element())
        uh = dolfinx.fem.Function(Vh, dtype=dtype)
        uh.interpolate(lambda x: x[1]**2-x[0])

        Wh = dolfinx.fem.functionspace(submesh, w.ufl_element())
        wh = dolfinx.fem.Function(Wh, dtype=dtype)
        wh.interpolate(lambda x: -3 * x[1]**3+x[0])
        form = dolfinx.fem.create_form(compiled_form, [], mesh, {}, {u: uh, w: wh}, {}, {submesh: parent_to_sub})
        assert np.isclose(mesh.comm.allreduce(dolfinx.fem.assemble_scalar(form), op=MPI.SUM), 1.5)

    # Create various meshes, that all uses this compiled form with a map from ufl
    # to dolfinx functions and constants
    for i in range(1, 4):
        create_and_integrate(i, compiled_form)


