# Copyright (C) 2024-2025 Jørgen S. Dokken
#
# This file is part of DOLFINx (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

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
    u = ufl.TestFunction(V)

    f_el = basix.ufl.element("Lagrange", "interval", 1, shape=(2,), dtype=real_type)
    submesh = ufl.Mesh(f_el)
    sub_el = basix.ufl.element("Lagrange", "interval", 3, dtype=real_type)
    V_sub = ufl.FunctionSpace(submesh, sub_el)

    w = ufl.Coefficient(V_sub)

    subdomain_id = 3
    J = ufl.inner(w, u) * ufl.ds(domain=domain, subdomain_id=subdomain_id)

    # Compile form using dolfinx.jit.ffcx_jit
    compiled_form = dolfinx.fem.compile_form(
        MPI.COMM_WORLD, J, form_compiler_options={"scalar_type": dtype}
    )

    def create_and_integrate(N, compiled_form):
        mesh = dolfinx.mesh.create_rectangle(
            MPI.COMM_WORLD,
            [np.array([0, 0]), np.array([2, 2])],
            [N, N],
            dolfinx.mesh.CellType.triangle,
            dtype=real_type,
        )
        assert mesh.ufl_domain().ufl_coordinate_element() == c_el

        facets = dolfinx.mesh.locate_entities_boundary(
            mesh, mesh.topology.dim - 1, lambda x: np.isclose(x[1], 2)
        )
        submesh, entity_map, _, _ = dolfinx.mesh.create_submesh(mesh, mesh.topology.dim - 1, facets)

        def g(x):
            return -3 * x[1] ** 3 + x[0]

        Vh = dolfinx.fem.functionspace(mesh, u.ufl_element())

        Wh = dolfinx.fem.functionspace(submesh, w.ufl_element())
        wh = dolfinx.fem.Function(Wh, dtype=dtype)
        wh.interpolate(g)

        facet_entities = dolfinx.fem.compute_integration_domains(
            dolfinx.fem.IntegralType.exterior_facet, mesh.topology, facets
        )
        subdomains = {dolfinx.fem.IntegralType.exterior_facet: [(subdomain_id, facet_entities)]}

        form = dolfinx.fem.create_form(
            compiled_form, [Vh], mesh, subdomains, {w: wh}, {}, [entity_map]
        )

        # Compute exact solution
        x = ufl.SpatialCoordinate(mesh)
        ff = dolfinx.mesh.meshtags(
            mesh, mesh.topology.dim - 1, facets, np.full(len(facets), subdomain_id, dtype=np.int32)
        )
        vh = ufl.TestFunction(Vh)
        ex_solution = dolfinx.fem.assemble_vector(
            dolfinx.fem.form(
                ufl.inner(g(x), vh)
                * ufl.ds(domain=mesh, subdomain_data=ff, subdomain_id=subdomain_id),
                dtype=dtype,
            )
        )
        ex_solution.scatter_reverse(dolfinx.la.InsertMode.add)
        bh = dolfinx.fem.assemble_vector(form)
        bh.scatter_reverse(dolfinx.la.InsertMode.add)
        tol = float(5e2 * np.finfo(dtype).resolution)
        np.testing.assert_allclose(ex_solution.array, bh.array, atol=tol)

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
def test_eliminated_data(dtype):
    """
    Test that mesh independent compilation handles the re-ordering of coefficients and constants
    when removed through differentiation
    """

    cell_name = "triangle"
    real_type = dtype(0).real.dtype
    c_el = basix.ufl.element("Lagrange", cell_name, 1, shape=(2,), dtype=real_type)
    domain = ufl.Mesh(c_el)
    el = basix.ufl.element("Lagrange", cell_name, 2, dtype=real_type)

    V = ufl.FunctionSpace(domain, el)

    c = ufl.Constant(domain)
    d = ufl.Constant(domain)
    u = ufl.Coefficient(V)
    v = ufl.Coefficient(V)

    J = (c * u**2 + d * v**2) * ufl.dx
    dv = ufl.conj(ufl.TestFunction(V))
    L = ufl.derivative(J, v, dv)

    # Compile form using dolfinx.jit.ffcx_jit
    compiled_form = dolfinx.fem.compile_form(
        MPI.COMM_WORLD, L, form_compiler_options={"scalar_type": dtype}
    )

    # Pack discrete data
    cell_type = dolfinx.mesh.to_type(cell_name)
    mesh = dolfinx.mesh.create_unit_square(
        MPI.COMM_WORLD, 5, 2, dtype=real_type, cell_type=cell_type
    )
    Vh = dolfinx.fem.functionspace(mesh, el)
    uh = dolfinx.fem.Function(Vh, dtype=dtype)
    uh.interpolate(lambda x: x[0])
    vh = dolfinx.fem.Function(Vh, dtype=dtype)
    vh.interpolate(lambda x: x[1])
    dh = dolfinx.fem.Constant(mesh, dtype(3.0))
    ch = dolfinx.fem.Constant(mesh, dtype(2.0))

    # Assemble discrete vector
    form = dolfinx.fem.create_form(compiled_form, [Vh], mesh, {}, {u: uh, v: vh}, {c: ch, d: dh})
    b = dolfinx.fem.assemble_vector(form)
    b.scatter_reverse(dolfinx.la.InsertMode.add)
    b.scatter_forward()

    # Compare to reference solution
    dvh = ufl.conj(ufl.TestFunction(Vh))
    exact_form = 2 * dh * vh * dvh * ufl.dx
    b_exact = dolfinx.fem.assemble_vector(dolfinx.fem.form(exact_form, dtype=dtype))
    b_exact.scatter_reverse(dolfinx.la.InsertMode.add)
    b_exact.scatter_forward()

    tol = np.finfo(dtype).resolution * 1e3
    np.testing.assert_allclose(b.array, b_exact.array, atol=tol)
