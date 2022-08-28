# # Scattering from a wire with perfectly matched layer condition
#
# Copyright (C) 2022 Michele Castriotta, Igor Baratta, Jørgen S. Dokken
#
# This demo is implemented in three files: one for the mesh
# generation with gmsh, one for the calculation of analytical efficiencies,
# and one for the variational forms and the solver. It illustrates how to:
#
# - Setup and solve Maxwell's equations for axisymmetric geometries
# - Implement (axisymmetric) perfectly matched layers
#
# ## Equations, problem definition and implementation
#
# First of all, let's import the modules that will be used:

# +
from scattnlay import scattnlay
import sys

try:
    import gmsh
except ModuleNotFoundError:
    print("This demo requires gmsh to be installed")
    sys.exit(0)
import numpy as np

try:
    import pyvista
    have_pyvista = True
except ModuleNotFoundError:
    print("pyvista and pyvistaqt are required to visualise the solution")
    have_pyvista = False

from functools import partial

import numpy as np
from mesh_sphere_axis import generate_mesh_sphere_axis
from scipy.special import jv

from dolfinx import fem, mesh, plot
from dolfinx.io import VTXWriter
from dolfinx.io.gmshio import model_to_mesh
from ufl import (FacetNormal, FiniteElement, Measure, MixedElement,
                 SpatialCoordinate, TestFunction, TrialFunction, as_matrix,
                 as_vector, conj, cross, det, grad, inner, inv, lhs, rhs, sqrt,
                 transpose)

from mpi4py import MPI
from petsc4py import PETSc

# -

# Since we want to solve time-harmonic Maxwell's equation, we need to
# specify that the demo should only be executed with DOLFINx complex mode,
# otherwise it would not work:

if not np.issubdtype(PETSc.ScalarType, np.complexfloating):
    print("Demo should only be executed with DOLFINx complex mode")
    exit(0)


# Now, let's consider a metallic sphere immersed in
# a background medium (e.g. vacuum or water) hit by a plane wave.
# We want to know what is the electric field scattered from the sphere.
# This problem is 3D, but can be simplified into many 2D problems
# by exploiting its axisymmetric nature. Let's see how.
#

def pml_coordinate(x, r, alpha, k0, radius_dom, radius_pml):

    return (x + 1j * alpha / k0 * x * (r - radius_dom) / (radius_pml * r))


def curl_axis(a, m, x):

    curl_r = -a[2].dx(1) - 1j * m / x[0] * a[1]
    curl_z = a[2] / x[0] + a[2].dx(0) + 1j * m / x[0] * a[0]
    curl_p = a[0].dx(1) - a[1].dx(0)

    return as_vector((curl_r, curl_z, curl_p))


def background_field_rz(theta, n_b, k0, m, x):

    k = k0 * n_b

    if m == 0:

        jv_prime = - jv(1, k * x[0] * np.sin(theta))

    else:

        jv_prime = 0.5 * (jv(m - 1, k * x[0] * np.sin(theta))
                          - jv(m + 1, k * x[0] * np.sin(theta)))

    a_r = (np.cos(theta) * np.exp(1j * k * x[1] * np.cos(theta))
           * (1j)**(-m + 1) * jv_prime)

    a_z = (np.sin(theta) * np.exp(1j * k * x[1] * np.cos(theta))
           * (1j)**-m * jv(m, k * x[0] * np.sin(theta)))

    return (a_r, a_z)


def background_field_p(theta, n_b, k0, m, x):

    k = k0 * n_b

    a_p = (np.cos(theta) / (k * x[0] * np.sin(theta))
           * np.exp(1j * k * x[1] * np.cos(theta)) * m
           * (1j)**(-m) * jv(m, k * x[0] * np.sin(theta)))

    return a_p


um = 1
nm = um * 10**-3

epsilon_0 = 8.8541878128 * 10**-12
mu_0 = 4 * np.pi * 10**-7

radius_sph = 0.025 * um
radius_dom = 0.200 * um
radius_pml = 0.025 * um

mesh_factor = 1

in_sph_size = mesh_factor * 2 * nm
on_sph_size = mesh_factor * 2 * nm
bkg_size = mesh_factor * 10 * nm
scatt_size = mesh_factor * 10 * nm
pml_size = mesh_factor * 10 * nm

# Tags for the subdomains
au_tag = 1
bkg_tag = 2
pml_tag = 3
scatt_tag = 4

model = generate_mesh_sphere_axis(
    radius_sph, radius_dom, radius_pml,
    in_sph_size, on_sph_size, bkg_size,
    scatt_size, pml_size,
    au_tag, bkg_tag, pml_tag, scatt_tag)

domain, cell_tags, facet_tags = model_to_mesh(
    model, MPI.COMM_WORLD, 0, gdim=2)

gmsh.finalize()
MPI.COMM_WORLD.barrier()

# Let's have a visual check of the mesh and of the subdomains
# by plotting them with PyVista:

if have_pyvista:
    topology, cell_types, geometry = plot.create_vtk_mesh(domain, 2)
    grid = pyvista.UnstructuredGrid(topology, cell_types, geometry)
    pyvista.set_jupyter_backend("pythreejs")
    plotter = pyvista.Plotter()
    num_local_cells = domain.topology.index_map(domain.topology.dim).size_local
    grid.cell_data["Marker"] = \
        cell_tags.values[cell_tags.indices < num_local_cells]
    grid.set_active_scalars("Marker")
    plotter.add_mesh(grid, show_edges=True)
    plotter.view_xy()
    if not pyvista.OFF_SCREEN:
        plotter.show()
    else:
        pyvista.start_xvfb()
        figure = plotter.screenshot("sphere_axis_mesh.png",
                                    window_size=[800, 800])

n_bkg = 1  # Background refractive index
eps_bkg = n_bkg**2  # Background relative permittivity

degree = 3
curl_el = FiniteElement("N1curl", domain.ufl_cell(), degree)
lagr_el = FiniteElement("Lagrange", domain.ufl_cell(), degree)
V = fem.FunctionSpace(domain, MixedElement(curl_el, lagr_el))

# Measures for subdomains
dx = Measure("dx", domain, subdomain_data=cell_tags,
             metadata={'quadrature_degree': 30})

dDom = dx((au_tag, bkg_tag))
dPml = dx(pml_tag)

eps_au = -1.0782 + 1j * 5.8089

D = fem.FunctionSpace(domain, ("DG", 0))
eps = fem.Function(D)
au_cells = cell_tags.find(au_tag)
bkg_cells = cell_tags.find(bkg_tag)
eps.x.array[au_cells] = np.full_like(
    au_cells, eps_au, dtype=np.complex128)
eps.x.array[bkg_cells] = np.full_like(bkg_cells, eps_bkg, dtype=np.complex128)
eps.x.scatter_forward()

wl0 = 0.4 * um  # Wavelength of the background field
k0 = 2 * np.pi / wl0  # Wavevector of the background field
deg = np.pi / 180
theta = 90 * deg  # Angle of incidence of the background field
m_list = [0, 1]


x = SpatialCoordinate(domain)
alpha = 1
r = sqrt(x[0]**2 + x[1]**2)

pml_coords = as_vector((
    pml_coordinate(
        x[0],
        r, alpha, k0, radius_dom, radius_pml),
    pml_coordinate(
        x[1],
        r, alpha, k0, radius_dom, radius_pml)))


def create_eps_mu(pml, x, eps_bkg, mu_bkg):

    J = grad(pml)

    # Transform the 2x2 Jacobian into a 3x3 matrix.
    J = as_matrix(((J[0, 0], J[0, 1], 0),
                   (J[1, 0], J[1, 1], 0),
                   (0, 0, pml[0] / x[0])))
    A = inv(J)
    eps_pml = det(J) * A * eps_bkg * transpose(A)
    mu_pml = det(J) * A * mu_bkg * transpose(A)
    return eps_pml, mu_pml


eps_pml, mu_pml = create_eps_mu(pml_coords, x, eps_bkg, 1)

V_dg = fem.VectorFunctionSpace(domain, ("DG", degree))

Esh_m_dg = fem.Function(V_dg)

# Vacuum impedance
Z0 = np.sqrt(mu_0 / epsilon_0)

Eh_m = fem.Function(V)

n = FacetNormal(domain)
n_3d = as_vector((n[0], n[1], 0))

# Intensity of the electromagnetic fields I0 = 0.5*E0**2/Z0
# E0 = np.sqrt(ax**2 + ay**2) = 1, see background_electric_field
I0 = 0.5 / Z0

# Geometrical cross section of the wire
gcs = np.pi * radius_sph**2

marker = fem.Function(D)
scatt_facets = facet_tags.find(scatt_tag)
incident_cells = mesh.compute_incident_entities(domain, scatt_facets,
                                                domain.topology.dim - 1,
                                                domain.topology.dim)

midpoints = mesh.compute_midpoints(
    domain, domain.topology.dim, incident_cells)
inner_cells = incident_cells[(midpoints[:, 0]**2
                              + midpoints[:, 1]**2) < (radius_dom)**2]

marker.x.array[inner_cells] = 1

# Define integration domain for the wire
dAu = dx(au_tag)

# Define integration facet for the scattering efficiency
dS = Measure("dS", domain, subdomain_data=facet_tags)

for m in m_list:
    Eb_m = fem.Function(V)

    # Definition of Trial and Test functions
    Es_m = TrialFunction(V)
    v_m = TestFunction(V)

    f_rz = partial(background_field_rz, theta, n_bkg, k0, m)
    f_p = partial(background_field_p, theta, n_bkg, k0, m)
    Eb_m.sub(0).interpolate(f_rz)
    Eb_m.sub(1).interpolate(f_p)

    curl_Es_m = curl_axis(Es_m, m, x)
    curl_v_m = curl_axis(v_m, m, x)

    F = - inner(curl_Es_m, curl_v_m) * x[0] * dDom \
        + eps * k0 ** 2 * inner(Es_m, v_m) * x[0] * dDom \
        + k0 ** 2 * (eps - eps_bkg) * inner(Eb_m, v_m) * x[0] * dDom \
        - inner(inv(mu_pml) * curl_Es_m, curl_v_m) * x[0] * dPml \
        + k0 ** 2 * inner(eps_pml * Es_m, v_m) * x[0] * dPml

    a, L = lhs(F), rhs(F)

    problem = fem.petsc.LinearProblem(a, L, bcs=[], petsc_options={
                                      "ksp_type": "preonly", "pc_type": "lu"})
    Esh_m = problem.solve()

    Esh_rz_m, Esh_p_m = Esh_m.split()

    Esh_m_dg.interpolate(Esh_rz_m)

    with VTXWriter(domain.comm, f"Esh_{m}.bp", Esh_m_dg) as f:
        f.write(0.0)

    Hsh_m = 1j * curl_axis(Esh_m, m, x) / Z0 / k0 / n_bkg

    Eh_m.x.array[:] = Eb_m.x.array[:] + Esh_m.x.array[:]

    # Quantities for the calculation of efficiencies

    if m == 0:
        P = np.pi * inner(cross(Esh_m, conj(Hsh_m)), n_3d) * marker
        Q = np.pi * eps_au.imag * k0 * (inner(Eh_m, Eh_m)) / Z0 / n_bkg

    else:  # we can improve it by using m_list instead of m, in that case use elif
        P += 2 * np.pi * inner(cross(Esh_m, conj(Hsh_m)), n_3d) * marker
        Q += 2 * np.pi * eps_au.imag * k0 * (inner(Eh_m, Eh_m)) / Z0 / n_bkg

# Normalized absorption efficiency
q_abs_fenics_proc = (fem.assemble_scalar(
    fem.form(Q * x[0] * dAu)) / gcs / I0).real
# Sum results from all MPI processes
q_abs_fenics = domain.comm.allreduce(q_abs_fenics_proc, op=MPI.SUM)

# Normalized scattering efficiency
q_sca_fenics_proc = (fem.assemble_scalar(
    fem.form((P('+') + P('-')) * x[0] * dS(scatt_tag))) / gcs / I0).real

# Sum results from all MPI processes
q_sca_fenics = domain.comm.allreduce(q_sca_fenics_proc, op=MPI.SUM)

# Extinction efficiency
q_ext_fenics = q_abs_fenics + q_sca_fenics

analyt_effs = scattnlay(np.array(
    [2 * np.pi * radius_sph / wl0 * n_bkg],
    dtype=np.complex128),
    np.array(
    [np.sqrt(eps_au) / n_bkg],
    dtype=np.complex128))

q_ext_analyt, q_sca_analyt, q_abs_analyt = analyt_effs[1:4]

# Error calculation
err_abs = np.abs(q_abs_analyt - q_abs_fenics) / q_abs_analyt
err_sca = np.abs(q_sca_analyt - q_sca_fenics) / q_sca_analyt
err_ext = np.abs(q_ext_analyt - q_ext_fenics) / q_ext_analyt

if MPI.COMM_WORLD.rank == 0:

    print()
    print(f"The analytical absorption efficiency is {q_abs_analyt}")
    print(f"The numerical absorption efficiency is {q_abs_fenics}")
    print(f"The error is {err_abs*100}%")
    print()
    print(f"The analytical scattering efficiency is {q_sca_analyt}")
    print(f"The numerical scattering efficiency is {q_sca_fenics}")
    print(f"The error is {err_sca*100}%")
    print()
    print(f"The analytical extinction efficiency is {q_ext_analyt}")
    print(f"The numerical extinction efficiency is {q_ext_fenics}")
    print(f"The error is {err_ext*100}%")
