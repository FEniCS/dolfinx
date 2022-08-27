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
from ufl import (FiniteElement, Measure, MixedElement,
                 SpatialCoordinate, TestFunction, TrialFunction,
                 as_matrix, as_vector, det, grad, inner, inv, lhs,
                 rhs, sqrt, transpose)

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
# a background medium (e.g. vacuum or water) hit by a plane wave. We want to know what is the electric field scattered from the sphere. This problem is 3D, but can be simplified into many 2D problems by exploiting its axisymmetric nature. Let's see how.
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
             metadata={'quadrature_degree': 20})

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
m = 0

Eb_m = fem.Function(V)
f_rz = partial(background_field_rz, theta, n_bkg, k0, m)
f_p = partial(background_field_p, theta, n_bkg, k0, m)
Eb_m.sub(0).interpolate(f_rz)
Eb_m.sub(1).interpolate(f_p)

# Definition of Trial and Test functions
Es_m = TrialFunction(V)
v_m = TestFunction(V)


x = SpatialCoordinate(domain)
alpha = 1
r = sqrt(x[0]**2 + x[1]**2)

pml_coords = as_vector((pml_coordinate(x[0], r, alpha, k0, radius_dom, radius_pml),
                        pml_coordinate(x[1], r, alpha, k0, radius_dom, radius_pml)))

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

# +
# Definition of the weak form

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

Esh_rz_m ,Esh_p_m = Esh_m.split()

V_dg = fem.VectorFunctionSpace(domain, ("DG", degree))
Esh_dg = fem.Function(V_dg)
Esh_dg.interpolate(Esh_rz_m)

with VTXWriter(domain.comm, "Esh.bp", Esh_dg) as f:
    f.write(0.0)

Hs_m = 1j * curl_axis(Esh_m, m, x)

E_m = fem.Function(V)
E_m.x.array[:] = Eb_m.x.array[:] + Esh_m.x.array[:]

