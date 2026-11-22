"""Finite-element space, material field, weak form, and linear solve."""

from functools import partial

import numpy as np
from petsc4py import PETSc

import ufl
from basix.ufl import element
from dolfinx import default_real_type, default_scalar_type, fem
from dolfinx.fem.petsc import LinearProblem

from fields import background_field, curl_2d
from pml import create_pml_tensors


def check_complex_mode():
    if not np.issubdtype(default_scalar_type, np.complexfloating):
        PETSc.Sys.Print("Demo should only be executed with DOLFINx complex mode")
        raise SystemExit(0)


def create_function_space(domain, degree: int):
    curl_el = element("N1curl", domain.basix_cell(), degree, dtype=default_real_type)
    return fem.functionspace(domain, curl_el)


def interpolate_background_field(V, problem_params):
    Eb = fem.Function(V)
    f = partial(background_field, problem_params.theta, problem_params.n_bkg, problem_params.k0)
    Eb.interpolate(f)
    return Eb


def create_permittivity(mesh_data, tags, problem_params):
    D = fem.functionspace(mesh_data.mesh, ("DG", 0))
    eps = fem.Function(D)

    au_cells = mesh_data.cell_tags.find(tags.au)
    bkg_cells = mesh_data.cell_tags.find(tags.bkg)

    eps.x.array[au_cells] = np.full_like(au_cells, problem_params.eps_au, dtype=eps.x.array.dtype)
    eps.x.array[bkg_cells] = np.full_like(bkg_cells, problem_params.eps_bkg, dtype=eps.x.array.dtype)
    eps.x.scatter_forward()

    return D, eps


def create_measures(mesh_data, tags):
    dx = ufl.Measure("dx", mesh_data.mesh, subdomain_data=mesh_data.cell_tags)
    dDom = dx((tags.au, tags.bkg))
    dPml_xy = dx(tags.pml)
    dPml_x = dx(tags.pml + 1)
    dPml_y = dx(tags.pml + 2)
    return dx, dDom, dPml_xy, dPml_x, dPml_y


def create_variational_forms(mesh_data, tags, mesh_params, problem_params, V, Eb, eps):
    Es = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    # Definition of 3D fields
    Es_3d = ufl.as_vector((Es[0], Es[1], 0))
    v_3d = ufl.as_vector((v[0], v[1], 0))

    dx, dDom, dPml_xy, dPml_x, dPml_y = create_measures(mesh_data, tags)
    eps_x, mu_x, eps_y, mu_y, eps_xy, mu_xy = create_pml_tensors(mesh_data.mesh, mesh_params, problem_params)

    k0 = problem_params.k0
    eps_bkg = problem_params.eps_bkg

    # Definition of the weak form
    F = (
        -ufl.inner(curl_2d(Es), curl_2d(v)) * dDom
        + eps * (k0**2) * ufl.inner(Es, v) * dDom
        + (k0**2) * (eps - eps_bkg) * ufl.inner(Eb, v) * dDom
        - ufl.inner(ufl.inv(mu_x) * curl_2d(Es), curl_2d(v)) * dPml_x
        - ufl.inner(ufl.inv(mu_y) * curl_2d(Es), curl_2d(v)) * dPml_y
        - ufl.inner(ufl.inv(mu_xy) * curl_2d(Es), curl_2d(v)) * dPml_xy
        + (k0**2) * ufl.inner(eps_x * Es_3d, v_3d) * dPml_x
        + (k0**2) * ufl.inner(eps_y * Es_3d, v_3d) * dPml_y
        + (k0**2) * ufl.inner(eps_xy * Es_3d, v_3d) * dPml_xy
    )

    return ufl.lhs(F), ufl.rhs(F), dx


def select_lu_backend(domain):
    # For factorisation prefer MUMPS, then superlu_dist, then default.
    petsc_sys = PETSc.Sys()  # type: ignore
    use_superlu = PETSc.IntType == np.int64

    if petsc_sys.hasExternalPackage("mumps") and not use_superlu:  # type: ignore
        return "mumps"
    if petsc_sys.hasExternalPackage("superlu_dist"):  # type: ignore
        return "superlu_dist"
    if domain.comm.size > 1:
        raise RuntimeError("This demo requires a parallel LU solver.")
    return "petsc"


def solve_scattered_field(a, L, domain):
    mat_factor_backend = select_lu_backend(domain)

    problem = LinearProblem(
        a,
        L,
        bcs=[],
        petsc_options_prefix="demo_pml_",
        petsc_options={
            "ksp_type": "preonly",
            "pc_type": "lu",
            "pc_factor_mat_solver_type": mat_factor_backend,
            "ksp_error_if_not_converged": True,
        },
    )

    Esh = problem.solve()
    assert isinstance(Esh, fem.Function)
    return Esh


def create_total_field(V, Eb, Esh):
    """$\mathbf{E}=\mathbf{E}_s+\mathbf{E}_b$ """
    E = fem.Function(V)
    E.x.array[:] = Eb.x.array[:] + Esh.x.array[:]
    return E
