# # Scattering from a wire with perfectly matched layer condition
#
# Copyright (C) 2022 Michele Castriotta, Igor Baratta, Jørgen S. Dokken
#
# This demo is implemented in three files: one for the mesh
# generation with gmsh, one for the calculation of analytical efficiencies,
# and one for the variational forms and the solver. It illustrates how to:
#
# - Use complex quantities in FEniCSx
# - Setup and solve Maxwell's equations
# - Implement (rectangular) perfectly matched layers
#
# ## Equations, problem definition and implementation
#
# First of all, let's import the modules that will be used:

# +
from re import M
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

import numpy as np
from mesh_sphere_axis import generate_mesh_sphere_axis
from scipy.special import jv

from dolfinx import fem, mesh, plot
from dolfinx.io import VTXWriter
from dolfinx.io.gmshio import model_to_mesh
from ufl import (FacetNormal, FiniteElement, MixedElement, Measure, SpatialCoordinate,
                 TestFunction, TrialFunction, algebra, as_matrix, as_vector,
                 conj, cross, det, grad, inner, inv, lhs, rhs, sqrt, transpose)

from mpi4py import MPI
from petsc4py import PETSc

# -

# Since we want to solve time-harmonic Maxwell's equation, we need to
# specify that the demo should only be executed with DOLFINx complex mode,
# otherwise it would not work:

if not np.issubdtype(PETSc.ScalarType, np.complexfloating):
    print("Demo should only be executed with DOLFINx complex mode")
    exit(0)


def pml_coordinates(x, r, alpha, k0, radius_dom, radius_pml):

    return (x + 1j * alpha / k0 * x * (r - radius_dom) / (radius_pml * r))


def curl_axis(v, x, m):

    curl_r = -v[2].dx(1) - 1j * m / x[0] * v[1]
    curl_z = v[2] / x[0] + v[2].dx(0) + 1j * m / x[0] * v[0]
    curl_p = v[0].dx(1) - v[1].dx(0)

    return as_vector((curl_r, curl_z, curl_p))


class BackgroundField:

    def __init__(self, theta, n_b, k0, m):
        self.theta = theta
        self.n_b = n_b
        self.k0 = k0
        self.m = m

    def eval(self, x):

        k = self.k0 * self.n_b

        if self.m == 0:

            jv_prime = - jv(self.m + 1, self.k * x[0] * np.sin(self.theta))

        else:

            jv_prime = 0.5 * (jv(self.m - 1, self.k * x[0] * np.sin(self.theta))
                              - jv(self.m + 1, self.k * x[0] * np.sin(self.theta)))

        ar = np.cos(self.theta) * np.exp(1j * self.k * x[1] * np.cos(self.theta)) * (1j)**(-self.m + 1) * jv_prime
        az = np.sin(self.theta) * np.exp(1j * self.k * x[1] * np.cos(self.theta)) * (1j)**-self.m * jv(self.m, self.k * x[0] * np.sin(self.theta))
        ap = np.cos(self.theta) / (self.k * x[0] * np.sin(self.theta)) * np.exp(1j * self.k * x[1]
                                                                 * np.cos(self.theta)) * self.m * (1j)**(-self.m) * jv(self.m, self.k * x[0] * np.sin(self.theta))

        return (ar, az, ap)


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
    au_tag, bkg_tag, scatt_tag, pml_tag)

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

wl0 = 0.4 * um  # Wavelength of the background field
n_bkg = 1  # Background refractive index
eps_bkg = n_bkg**2  # Background relative permittivity
k0 = 2 * np.pi / wl0  # Wavevector of the background field
deg = np.pi / 180
theta = 0 * deg  # Angle of incidence of the background field
m = 0

degree = 3
curl_el = FiniteElement("N1curl", domain.ufl_cell(), degree)
lagr_el = FiniteElement("Lagrange", domain.ufl_cell(), degree)
V = fem.FunctionSpace(domain, MixedElement([curl_el, lagr_el]))

Eb = fem.Function(V)
f = BackgroundField(theta, n_bkg, k0, m)
Eb.interpolate(f.eval)

# Definition of Trial and Test functions
Es = TrialFunction(V)
v = TestFunction(V)

# Measures for subdomains
dx = Measure("dx", domain, subdomain_data=cell_tags)
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

x = SpatialCoordinate(domain)
alpha = 1
r = sqrt(x[0]**2 + x[1]**2)

pml_coords = as_vector((pml_coordinates(x[0], r, alpha, k0, radius_dom, radius_pml),
                        pml_coordinates(x[1], r, alpha, k0, radius_dom, radius_pml)))

def create_eps_mu(pml, r):

    J = grad(pml)

    # Transform the 2x2 Jacobian into a 3x3 matrix.
    J = as_matrix(((J[0, 0], J[0, 1], 0),
                   (J[1, 0], J[1, 1], 0),
                   (0, 0, pml[0]/r)))

    A = inv(J)
    eps = det(J) * A * eps_bkg * transpose(A)
    mu = det(J) * A * 1 * transpose(A)
    return eps, mu

eps_pml, mu_pml = create_eps_mu(pml_coords, r)

# +
# Definition of the weak form

F = - inner(curl_axis(Es), curl_axis(v)) * x[0] * dDom \
    + eps * k0 ** 2 * inner(Es, v) * x[0] * dDom \
    + k0 ** 2 * (eps - eps_bkg) * inner(Eb, v) * x[0] * dDom \
    - inner(inv(mu_pml) * curl_axis(Es), curl_axis(v)) * x[0] *dPml \
    + eps_bkg * k0 ** 2 * inner(eps * Es, v) * x[0] *dPml \

a, L = lhs(F), rhs(F)

problem = fem.petsc.LinearProblem(a, L, bcs=[], petsc_options={
                                  "ksp_type": "preonly", "pc_type": "lu"})
Esh = problem.solve()
# -