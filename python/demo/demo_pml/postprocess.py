"""Absorption, scattering, extinction efficiencies, and error reporting."""

import numpy as np
from mpi4py import MPI
from petsc4py import PETSc

import ufl
from dolfinx import fem, mesh

from fields import calculate_analytical_efficiencies, curl_2d


def calc_Q(E, eps_au, k0, Z0, n_bkg):
    """
    Absorption density used for q_abs:
        Q = 0.5 * Im(eps_au) * k0 * |E|^2 / (Z0 * n_bkg)
    Returns a UFL scalar expression.
    """
    E_3d = ufl.as_vector((E[0], E[1], 0))
    Q = 0.5 * eps_au.imag * k0 * ufl.inner(E_3d, E_3d) / (Z0 * n_bkg)
    return Q


def calc_P(Esh, k0, Z0, n_bkg, mesh_data, scatt_tag, radius_scatt, D, tdim):
    """
    Scattered Poynting flux (normal component) for q_sca:
        H_s = -j * curl(E_s)/(Z0 k0 n_bkg)
        P   = 0.5 * Re((E_s x H_s*) . n3d)
    Builds the inner-ring marker internally.
    Returns a UFL scalar expression.
    """
    n2d = ufl.FacetNormal(mesh_data.mesh)
    n3d = ufl.as_vector((n2d[0], n2d[1], 0))
    Esh_3d = ufl.as_vector((Esh[0], Esh[1], 0))
    Hsh_3d = -1j * curl_2d(Esh) / (Z0 * k0 * n_bkg)
    Sn = 0.5 * ufl.inner(ufl.cross(Esh_3d, ufl.conj(Hsh_3d)), n3d)

    # Create a marker for the integration boundary for the scattering efficiency.
    marker = fem.Function(D)
    scatt_facets = mesh_data.facet_tags.find(scatt_tag)
    incident_cells = mesh.compute_incident_entities(
        mesh_data.mesh.topology, scatt_facets, tdim - 1, tdim
    )
    mesh_data.mesh.topology.create_connectivity(tdim, tdim)
    midpoints = mesh.compute_midpoints(mesh_data.mesh, tdim, incident_cells)
    inner_cells = incident_cells[(midpoints[:, 0] ** 2 + midpoints[:, 1] ** 2) < (radius_scatt) ** 2]
    marker.x.array[inner_cells] = 1

    return Sn * marker


def compute_efficiencies(mesh_data, tags, mesh_params, problem_params, constants, D, dx, Esh, E):
    q_abs_analyt, q_sca_analyt, q_ext_analyt = calculate_analytical_efficiencies(
        problem_params.eps_au,
        problem_params.n_bkg,
        problem_params.wl0,
        mesh_params.radius_wire,
    )
    # Vacuum impedance
    Z0 = np.sqrt(constants.mu_0 / constants.epsilon_0)
    # Intensity of the electromagnetic fields I0 = 0.5*E0**2/Z0
    # E0 = np.sqrt(ax**2 + ay**2) = 1, see background_electric_field
    I0 = 0.5 / Z0
    gcs = 2 * mesh_params.radius_wire # Geometrical cross section of the wire
    tdim = mesh_data.mesh.topology.dim

    P = calc_P(
        Esh,
        problem_params.k0,
        Z0,
        problem_params.n_bkg,
        mesh_data,
        tags.scatt,
        mesh_params.radius_scatt,
        D,
        tdim,
    )
    # Quantities for the calculation of efficiencies
    Q = calc_Q(E, problem_params.eps_au, problem_params.k0, Z0, problem_params.n_bkg)

    # Normalized absorption efficiency.
    dAu = dx(tags.au)
    q_abs_fenics_proc = (fem.assemble_scalar(fem.form(Q * dAu)) / (gcs * I0)).real
    q_abs_fenics = mesh_data.mesh.comm.allreduce(q_abs_fenics_proc, op=MPI.SUM)

    # Normalized scattering efficiency.
    dS = ufl.Measure("dS", mesh_data.mesh, subdomain_data=mesh_data.facet_tags)
    q_sca_fenics_proc = (
        fem.assemble_scalar(fem.form((P("+") + P("-")) * dS(tags.scatt))) / (gcs * I0)
    ).real
    q_sca_fenics = mesh_data.mesh.comm.allreduce(q_sca_fenics_proc, op=MPI.SUM)

    # Extinction efficiency.
    q_ext_fenics = q_abs_fenics + q_sca_fenics

    err_abs = np.abs(q_abs_analyt - q_abs_fenics) / q_abs_analyt
    err_sca = np.abs(q_sca_analyt - q_sca_fenics) / q_sca_analyt
    err_ext = np.abs(q_ext_analyt - q_ext_fenics) / q_ext_analyt

    return {
        "analytical": (q_abs_analyt, q_sca_analyt, q_ext_analyt),
        "numerical": (q_abs_fenics, q_sca_fenics, q_ext_fenics),
        "errors": (err_abs, err_sca, err_ext),
    }


def print_and_check_efficiencies(results):
    q_abs_analyt, q_sca_analyt, q_ext_analyt = results["analytical"]
    q_abs_fenics, q_sca_fenics, q_ext_fenics = results["numerical"]
    err_abs, err_sca, err_ext = results["errors"]

    PETSc.Sys.Print(
        f"Analytical  : Q_abs={q_abs_analyt:.6f}, Q_sca={q_sca_analyt:.6f}, Q_ext={q_ext_analyt:.6f}"
    )
    PETSc.Sys.Print(
        f"Numerical   : Q_abs={q_abs_fenics:.6f},      Q_sca={q_sca_fenics:.6f},      Q_ext={q_ext_fenics:.6f}"
    )
    PETSc.Sys.Print(
        f"Error is : Q_abs = {err_abs * 100}%, Q_sca={err_sca * 100}%, Q_ext={err_ext * 100}%"
    )

    assert err_abs < 0.01, "Error in absorption efficiency is too large"
    # assert err_sca < 0.01
    assert err_ext < 0.01, "Error in extinction efficiency is too large"
