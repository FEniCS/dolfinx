# # Scattering from a sphere in the axisymmetric formulation
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
from functools import partial

import numpy as np
from mesh_sphere_axis import generate_mesh_sphere_axis
from scipy.special import jv, jvp

import ufl
from dolfinx import fem, io, mesh, plot

from mpi4py import MPI
from petsc4py import PETSc

try:
    from dolfinx.io import VTXWriter
    has_vtx = True
except ImportError:
    print("VTXWriter not available, solution won't be saved")
    has_vtx = False

try:
    import gmsh
except ModuleNotFoundError:
    print("This demo requires gmsh to be installed")
    sys.exit(0)

try:
    import pyvista
    have_pyvista = True
except ModuleNotFoundError:
    print("pyvista and pyvistaqt are required to visualise the solution")
    have_pyvista = False
# -

# Since we want to solve the time-harmonic version of Maxwell's equation, we need to solve a
# complex-valued PDE, and therefore need to use PETSc compiled with complex numbers.

if not np.issubdtype(PETSc.ScalarType, np.complexfloating):
    print("Demo should only be executed with DOLFINx complex mode")
    exit(0)


# Now, let's formulate our problem.
# Let's consider a metallic sphere immersed in
# a background medium (e.g. vacuum or water) and hit by a plane wave.
# We want to calculate the scattered electric field.
# Even though the problem is three-dimensional,
# we can simplify it into many two-dimensional problems
# by exploiting axisymmetry. To verify this, let's consider
# the weak form of the problem with PML:
#
# $$
# \begin{align}
# &\int_{\Omega_m\cup\Omega_b}-(\nabla \times \mathbf{E}_s)
# \cdot (\nabla \times \bar{\mathbf{v}})+\varepsilon_{r} k_{0}^{2}
# \mathbf{E}_s \cdot \bar{\mathbf{v}}+k_{0}^{2}\left(\varepsilon_{r}
# -\varepsilon_b\right)\mathbf{E}_b \cdot \bar{\mathbf{v}}~\mathrm{d} x\\
# +&\int_{\Omega_{pml}}\left[\boldsymbol{\mu}^{-1}_{pml} \nabla \times \mathbf{E}_s
# \right]\cdot \nabla \times \bar{\mathbf{v}}-k_{0}^{2}
# \left[\boldsymbol{\varepsilon}_{pml} \mathbf{E}_s \right]\cdot
# \bar{\mathbf{v}}~ d x=0
# \end{align}
# $$
#
# Since we want to exploit the axisymmetry of the problem,
# we can use cylindrical coordinates as a more appropriate
# reference system:
#
# $$
# \begin{align}
# &\int_{\Omega_{cs}}\int_{0}^{2\pi}-(\nabla \times \mathbf{E}_s)
# \cdot (\nabla \times \bar{\mathbf{v}})+\varepsilon_{r} k_{0}^{2}
# \mathbf{E}_s \cdot \bar{\mathbf{v}}+k_{0}^{2}\left(\varepsilon_{r}
# -\varepsilon_b\right)\mathbf{E}_b \cdot
# \bar{\mathbf{v}}~ \rho d\rho dz d \phi\\
# +&\int_{\Omega_{cs}}
# \int_{0}^{2\pi}\left[\boldsymbol{\mu}^{-1}_{pml} \nabla \times \mathbf{E}_s
# \right]\cdot \nabla \times \bar{\mathbf{v}}-k_{0}^{2}
# \left[\boldsymbol{\varepsilon}_{pml} \mathbf{E}_s \right]\cdot
# \bar{\mathbf{v}}~ \rho d\rho dz d \phi=0
# \end{align}
# $$
#
# Let's now expand $\mathbf{E}_s$,
# $\mathbf{E}_b$ and $\bar{\mathbf{v}}$ in cylindrical harmonics:
#
# $$
# \begin{align}
# & \mathbf{E}_s(\rho, z, \phi) = \sum_m\mathbf{E}^{(m)}_s(\rho, z)e^{-jm\phi} \\
# & \mathbf{E}_b(\rho, z, \phi) = \sum_m\mathbf{E}^{(m)}_b(\rho, z)e^{-jm\phi} \\
# & \bar{\mathbf{v}}(\rho, z, \phi) =
# \sum_m\bar{\mathbf{v}}^{(m)}(\rho, z)e^{+jm\phi}\\
# \end{align}
# $$
#
# The curl operator $\nabla\times$ in cylindrical coordinates becomes:
#
# $$
# \begin{aligned}
# \nabla \times \mathbf{a}=
# \sum_{m}\left(\nabla \times \mathbf{a}^{(m)}\right) e^{-j m \phi}
# \end{aligned}
# $$
#
# with:
#
# $$
# \begin{align}
# \left(\nabla \times \mathbf{a}^{(m)}\right) = &\left[\hat{\rho}
# \left(-\frac{\partial a_{\phi}^{(m)}}{\partial z}
# -j\frac{m}{\rho} a_{z}^{(m)}\right)+\\ \hat{\phi}
# \left(\frac{\partial a_{\rho}^{(m)}}{\partial z}
# -\frac{\partial a_{z}^{(m)}}{\partial \rho}\right)+\right.\\
# &\left.+\hat{z}\left(\frac{a_{\phi}^{(m)}}{\rho}
# +\frac{\partial a_{\phi}^{(m)}}{\partial \rho}
# +j \frac{m}{\rho} a_{\rho}^{(m)}\right)\right]
# \end{align}
# $$
#
# By implementing these formula in our weak form, and by assuming an
# axisymmetric permittivity $\varepsilon(\rho, z)$, we can write:
#
# $$
# \begin{align}
# \sum_{n, m}\int_{\Omega_{cs}}&-(\nabla \times \mathbf{E}^{(m)}_s)
# \cdot (\nabla \times \bar{\mathbf{v}}^{(m)})+\varepsilon_{r} k_{0}^{2}
# \mathbf{E}^{(m)}_s \cdot \bar{\mathbf{v}}^{(m)}+k_{0}^{2}
# \left(\varepsilon_{r}
# -\varepsilon_b\right)\mathbf{E}^{(m)}_b \cdot \bar{\mathbf{v}}^{(m)}\\
# &+\left(\boldsymbol{\mu}^{-1}_{pml} \nabla \times \mathbf{E}^{(m)}_s
# \right)\cdot \nabla \times \bar{\mathbf{v}}^{(m)}-k_{0}^{2}
# \left(\boldsymbol{\varepsilon}_{pml} \mathbf{E}^{(m)}_s \right)\cdot
# \bar{\mathbf{v}}^{(m)}~ \rho d\rho dz \int_{0}^{2 \pi} e^{-i(m-n) \phi}
# d \phi=0
# \end{align}
# $$
#
# For the last integral, we have that:
#
# $$
# \int_{0}^{2 \pi} e^{-i(m-n) \phi}d \phi=
# \begin{cases}
# \begin{align}
# 2\pi ~~~&\textrm{if } m=n\\
# 0 ~~~&\textrm{if }m\neq n\\
# \end{align}
# \end{cases}
# $$
#
# We can therefore consider only the case $m = n$ and simplify the summation
# in the following way:
#
# $$
# \begin{align}
# \sum_{m}\int_{\Omega_{cs}}&-(\nabla \times \mathbf{E}^{(m)}_s)
# \cdot (\nabla \times \bar{\mathbf{v}}^{(m)})+\varepsilon_{r} k_{0}^{2}
# \mathbf{E}^{(m)}_s \cdot \bar{\mathbf{v}}^{(m)}
# +k_{0}^{2}\left(\varepsilon_{r}
# -\varepsilon_b\right)\mathbf{E}^{(m)}_b \cdot \bar{\mathbf{v}}^{(m)}\\
# &+\left(\boldsymbol{\mu}^{-1}_{pml} \nabla \times \mathbf{E}^{(m)}_s
# \right)\cdot \nabla \times \bar{\mathbf{v}}^{(m)}-k_{0}^{2}
# \left(\boldsymbol{\varepsilon}_{pml} \mathbf{E}^{(m)}_s \right)\cdot
# \bar{\mathbf{v}}^{(m)}~ \rho d\rho dz =0
# \end{align}
# $$
#
# What we have just obtained are multiple weak forms corresponding to
# different cylindrical harmonics propagating independently.
# Let's now solve it in DOLFINx.
# As a first step we can define the function for the $\nabla\times$
# operator in cylindrical coordinates:

def curl_axis(a, m: int, rho):

    curl_r = -a[2].dx(1) - 1j * m / rho * a[1]
    curl_z = a[2] / rho + a[2].dx(0) + 1j * m / rho * a[0]
    curl_p = a[0].dx(1) - a[1].dx(0)

    return ufl.as_vector((curl_r, curl_z, curl_p))


# Then we need to define the analytical formula for the background field.
# We will consider a TMz polarized plane wave, that is a plane wave with
# the magnetic field normal to our reference plane
#
# For a TMz polarization, we can write the cylindrical harmonics $\mathbf{E}^{(m)}_b$
# of the background field in this way
# ([Wait 1955](https://doi.org/10.1139/p55-024)):
#
# $$
# \begin{align}
# \mathbf{E}^{(m)}_b = &\hat{\rho} \left(E_{0} \cos \theta
# e^{j k z \cos \theta} j^{-m+1} J_{m}^{\prime}\left(k_{0} \rho \sin
# \theta\right)\right)\\
# +&\hat{z} \left(E_{0} \sin \theta e^{j k z \cos \theta}j^{-m} J_{m}
# \left(k \rho \sin \theta\right)\right)\\
# +&\hat{\phi} \left(\frac{E_{0} \cos \theta}{k \rho \sin \theta}
# e^{j k z \cos \theta} mj^{-m} J_{m}\left(k \rho \sin \theta\right)\right)
# \end{align}
# $$
#
# with $k = 2\pi n_b/\lambda = k_0n_b$ being the wavevector, $\theta$ being
# the angle between $\mathbf{E}_b$ and $\hat{\rho}$, and $J_m$
# representing the $m$-th order Bessel function of first kind and
# $J_{m}^{\prime}$ its derivative.
# In DOLFINx, we can implement these functions in this way:

# +
def background_field_rz(theta: float, n_bkg: float, k0: float, m: int, x):

    k = k0 * n_bkg

    a_r = (np.cos(theta) * np.exp(1j * k * x[1] * np.cos(theta))
           * (1j)**(-m + 1) * jvp(m, k * x[0] * np.sin(theta), 1))

    a_z = (np.sin(theta) * np.exp(1j * k * x[1] * np.cos(theta))
           * (1j)**-m * jv(m, k * x[0] * np.sin(theta)))

    return (a_r, a_z)


def background_field_p(theta: float, n_bkg: float, k0: float, m: int, x):

    k = k0 * n_bkg

    a_p = (np.cos(theta) / (k * x[0] * np.sin(theta))
           * np.exp(1j * k * x[1] * np.cos(theta)) * m
           * (1j)**(-m) * jv(m, k * x[0] * np.sin(theta)))

    return a_p


# -

# PML can be implemented in a spherical shell surrounding the
# background domain. We can use the following complex coordinate
# transformation for PML:
#
# $$
# \begin{align}
# & \rho^{\prime} = \rho\left[1 +j \alpha/k_0 \left(\frac{r
# - r_{dom}}{r~r_{pml}}\right)\right] \\
# & z^{\prime} = z\left[1 +j \alpha/k_0 \left(\frac{r
# - r_{dom}}{r~r_{pml}}\right)\right] \\
# & \phi^{\prime} = \phi \\
# \end{align}
# $$
#
# with $\alpha$ tuning the absorption inside the PML,
# and $r = \sqrt{\rho^2 + z^2}$.
# This coordinate transformation has the following jacobian:
#
# $$
# \mathbf{J}=\mathbf{A}^{-1}= \nabla\boldsymbol{\rho}^
# \prime(\boldsymbol{\rho}) =
# \left[\begin{array}{ccc}
# \frac{\partial \rho^{\prime}}{\partial \rho} &
# \frac{\partial z^{\prime}}{\partial \rho} &
# 0 \\
# \frac{\partial \rho^{\prime}}{\partial z} &
# \frac{\partial z^{\prime}}{\partial z} &
# 0 \\
# 0 &
# 0 &
# \frac{\rho^\prime}{\rho}\frac{\partial \phi^{\prime}}{\partial \phi}
# \end{array}\right]=\left[\begin{array}{ccc}
# J_{11} & J_{12} & 0 \\
# J_{21} & J_{22} & 0 \\
# 0 & 0 & J_{33}
# \end{array}\right]
# $$
#
# which we can use to calculate ${\boldsymbol{\varepsilon}_{pml}}$ and
# ${\boldsymbol{\mu}_{pml}}$:
#
# $$
# \begin{align}
# & {\boldsymbol{\varepsilon}_{pml}} =
# A^{-1} \mathbf{A} {\boldsymbol{\varepsilon}_b}\mathbf{A}^{T}\\
# & {\boldsymbol{\mu}_{pml}} =
# A^{-1} \mathbf{A} {\boldsymbol{\mu}_b}\mathbf{A}^{T}
# \end{align}
# $$
#
# For doing these calculations, we define the
# `pml_coordinate` and `create_mu_eps` functions:

# +
def pml_coordinate(
        x, r, alpha: float, k0: float, radius_dom: float, radius_pml: float):

    return (x + 1j * alpha / k0 * x * (r - radius_dom) / (radius_pml * r))


def create_eps_mu(pml, rho, eps_bkg, mu_bkg):

    J = ufl.grad(pml)

    # Transform the 2x2 Jacobian into a 3x3 matrix.
    J = ufl.as_matrix(((J[0, 0], J[0, 1], 0),
                       (J[1, 0], J[1, 1], 0),
                       (0, 0, pml[0] / rho)))

    A = ufl.inv(J)
    eps_pml = ufl.det(J) * A * eps_bkg * ufl.transpose(A)
    mu_pml = ufl.det(J) * A * mu_bkg * ufl.transpose(A)
    return eps_pml, mu_pml


# -

# We can now define some constants and geometrical parameters,
# and then we can generate the mesh with Gmsh, by using the
# function `generate_mesh_sphere_axis` in `mesh_sphere_axis.py`:

# +
# Constants
epsilon_0 = 8.8541878128 * 10**-12  # Vacuum permittivity
mu_0 = 4 * np.pi * 10**-7  # Vacuum permeability
Z0 = np.sqrt(mu_0 / epsilon_0)  # Vacuum impedance

# Radius of the sphere
radius_sph = 0.025

# Radius of the domain
radius_dom = 1

# Radius of the boundary where scattering efficiency
# is calculated
radius_scatt = 0.4 * radius_dom

# Radius of the PML shell
radius_pml = 0.25

# Mesh sizes
mesh_factor = 1
in_sph_size = mesh_factor * 2.0e-3
on_sph_size = mesh_factor * 2.0e-3
scatt_size = mesh_factor * 60.0e-3
pml_size = mesh_factor * 40.0e-3

# Tags for the subdomains
au_tag = 1
bkg_tag = 2
pml_tag = 3
scatt_tag = 4

model = None
gmsh.initialize(sys.argv)
if MPI.COMM_WORLD.rank == 0:

    # Mesh generation
    model = generate_mesh_sphere_axis(
        radius_sph, radius_scatt, radius_dom, radius_pml,
        in_sph_size, on_sph_size, scatt_size, pml_size,
        au_tag, bkg_tag, pml_tag, scatt_tag)

model = MPI.COMM_WORLD.bcast(model, root=0)
domain, cell_tags, facet_tags = io.gmshio.model_to_mesh(
    model, MPI.COMM_WORLD, 0, gdim=2)

gmsh.finalize()
MPI.COMM_WORLD.barrier()
# -

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
                                    window_size=[500, 500])

# We can now define our function space. For the $\hat{\rho}$ and $\hat{z}$
# components of the electric field, we will use Nedelec elements,
# while for the $\hat{\phi}$ components we will use Lagrange elements:

degree = 3
curl_el = ufl.FiniteElement("N1curl", domain.ufl_cell(), degree)
lagr_el = ufl.FiniteElement("Lagrange", domain.ufl_cell(), degree)
V = fem.FunctionSpace(domain, ufl.MixedElement([curl_el, lagr_el]))

# The integration domains of our problem are the following:

# +
# Measures for subdomains
dx = ufl.Measure("dx", domain, subdomain_data=cell_tags,
                 metadata={'quadrature_degree': 5})

dDom = dx((au_tag, bkg_tag))
dPml = dx(pml_tag)
# -

# Let's now add the $\varepsilon_r$ function:

# +
n_bkg = 1  # Background refractive index
eps_bkg = n_bkg**2  # Background relative permittivity
eps_au = -1.0782 + 1j * 5.8089

D = fem.FunctionSpace(domain, ("DG", 0))
eps = fem.Function(D)
au_cells = cell_tags.find(au_tag)
bkg_cells = cell_tags.find(bkg_tag)
eps.x.array[au_cells] = np.full_like(
    au_cells, eps_au, dtype=np.complex128)
eps.x.array[bkg_cells] = np.full_like(bkg_cells, eps_bkg, dtype=np.complex128)
eps.x.scatter_forward()
# -

# For the background field, we need to specify the wavelength
# (and the wavevector), the angle of incidence, the harmonic numbers
# and the intensity. Therefore, we can write:

wl0 = 0.4  # Wavelength of the background field
k0 = 2 * np.pi / wl0  # Wavevector of the background field
theta = np.pi / 4  # Angle of incidence of the background field
m_list = [0, 1]  # list of harmonics
I0 = 0.5 * n_bkg / Z0  # Intensity


# with `m_list` being a list containing the harmonic
# numbers we want to solve the problem for. In general,
# we would need to solve for $m\in \mathbb{Z}$. However,
# for subwavelength structure (as our sphere),
# we can limit the calculation to few harmonic numbers, i.e.
# $m = -1, 0, 1$. Besides, we have that:
#
# \begin{aligned}
# &J_{-m}=(-1)^m J_m \\
# &J_{-m}^{\prime}=(-1)^m J_m^{\prime} \\
# &j^{-m}=(-1)^m j^m
# \end{aligned}
#
# and therefore:
#
# \begin{aligned}
# &E_{b, \rho}^{(m)}=E_{b, \rho}^{(-m)} \\
# &E_{b, \phi}^{(m)}=-E_{b, \phi}^{(-m)} \\
# &E_{b, z}^{(m)}=E_{b, z}^{(-m)}
# \end{aligned}
#
# In light of this, we can just solve the problem for $m\geq 0$.
#

# Let's now define `eps_pml` and `mu_pml`:

# +
rho, z = ufl.SpatialCoordinate(domain)
alpha = 5
r = ufl.sqrt(rho**2 + z**2)

pml_coords = ufl.as_vector((
    pml_coordinate(rho, r, alpha, k0, radius_dom, radius_pml),
    pml_coordinate(z, r, alpha, k0, radius_dom, radius_pml)))

eps_pml, mu_pml = create_eps_mu(pml_coords, rho, eps_bkg, 1)
# -

# We can now define other objects that will be used inside our
# solver loop:

# +
# Total field
Eh_m = fem.Function(V)
Esh = fem.Function(V)

n = ufl.FacetNormal(domain)
n_3d = ufl.as_vector((n[0], n[1], 0))

# Geometrical cross section of the sphere, for efficiency calculation
gcs = np.pi * radius_sph**2

# Marker functions for the scattering efficiency integral
marker = fem.Function(D)
scatt_facets = facet_tags.find(scatt_tag)
incident_cells = mesh.compute_incident_entities(domain, scatt_facets,
                                                domain.topology.dim - 1,
                                                domain.topology.dim)
midpoints = mesh.compute_midpoints(
    domain, domain.topology.dim, incident_cells)
inner_cells = incident_cells[(midpoints[:, 0]**2
                              + midpoints[:, 1]**2) < (radius_scatt)**2]
marker.x.array[inner_cells] = 1

# Define integration domain for the gold sphere
dAu = dx(au_tag)

# Define integration facet for the scattering efficiency
dS = ufl.Measure("dS", domain, subdomain_data=facet_tags)
# -

# We also specify a variable `phi`, corresponding to the $\phi$
# angle of the cylindrical coordinate system that we will use to
# post-process the field and save it along the plane at
# $\phi = \pi/4$. In particular, the scattered field needs to be
# transformed for $m\neq 0$ in the following way:
#
# \begin{aligned}
# &E_{s, \rho}^{(m)}(\phi)=E_{s, \rho}^{(m)}(e^{-jm\phi}+e^{jm\phi}) \\
# &E_{s, \phi}^{(m)}(\phi)=E_{s, \phi}^{(m)}(e^{-jm\phi}-e^{jm\phi}) \\
# &E_{s, z}^{(m)}(\phi)=E_{s, z}^{(m)}(e^{-jm\phi}+e^{jm\phi})
# \end{aligned}
#
# For this reason, we also add a `phase` constant for the above phase term:

# +
phi = np.pi / 4

# Initialize phase term
phase = fem.Constant(domain, PETSc.ScalarType(np.exp(1j * 0 * phi)))
# -

# Let's now solve the problem:

for m in m_list:

    # Definition of Trial and Test functions
    Es_m = ufl.TrialFunction(V)
    v_m = ufl.TestFunction(V)

    # Background field
    Eb_m = fem.Function(V)
    f_rz = partial(background_field_rz, theta, n_bkg, k0, m)
    f_p = partial(background_field_p, theta, n_bkg, k0, m)
    Eb_m.sub(0).interpolate(f_rz)
    Eb_m.sub(1).interpolate(f_p)

    curl_Es_m = curl_axis(Es_m, m, rho)
    curl_v_m = curl_axis(v_m, m, rho)

    F = - ufl.inner(curl_Es_m, curl_v_m) * rho * dDom \
        + eps * k0 ** 2 * ufl.inner(Es_m, v_m) * rho * dDom \
        + k0 ** 2 * (eps - eps_bkg) * ufl.inner(Eb_m, v_m) * rho * dDom \
        - ufl.inner(ufl.inv(mu_pml) * curl_Es_m, curl_v_m) * rho * dPml \
        + k0 ** 2 * ufl.inner(eps_pml * Es_m, v_m) * rho * dPml

    a, L = ufl.lhs(F), ufl.rhs(F)

    problem = fem.petsc.LinearProblem(a, L, bcs=[], petsc_options={
                                      "ksp_type": "preonly", "pc_type": "lu"})
    Esh_m = problem.solve()

    # Scattered magnetic field
    Hsh_m = -1j * curl_axis(Esh_m, m, rho) / (Z0 * k0)

    # Total electric field
    Eh_m.x.array[:] = Eb_m.x.array[:] + Esh_m.x.array[:]

# We can add our solution to the total scattered field, which also includes the transformation to the $\phi$ plane:

    phase.value = np.exp(-1j * m * phi)

    rotate_to_phi = ufl.as_vector(
        (phase + ufl.conj(phase),
         phase + ufl.conj(phase),
         phase - ufl.conj(phase)))

    if m == 0:  # initialize and do not transform

        Esh = Esh_m

    elif m == m_list[0]:  # initialize and transform

        Esh = ufl.elem_mult(Esh_m, rotate_to_phi)

    else:  # transform

        Esh += ufl.elem_mult(Esh_m, rotate_to_phi)

# In order to check whether our calculation are correct, we can use
# as reference quantities the absorption and scattering efficiencies:

# +
    # Efficiencies calculation

    if m == 0:  # initialize and do not add 2 factor
        P = np.pi * \
            ufl.inner(-ufl.cross(Esh_m, ufl.conj(Hsh_m)), n_3d) * marker
        Q = np.pi * eps_au.imag * k0 * (ufl.inner(Eh_m, Eh_m)) / Z0

        q_abs_fenics_proc = (fem.assemble_scalar(
                             fem.form(Q * rho * dAu)) / gcs / I0).real
        q_sca_fenics_proc = (fem.assemble_scalar(
                             fem.form((P('+') + P('-')) * rho * dS(scatt_tag))) / gcs / I0).real
        q_abs_fenics = domain.comm.allreduce(q_abs_fenics_proc, op=MPI.SUM)
        q_sca_fenics = domain.comm.allreduce(q_sca_fenics_proc, op=MPI.SUM)

    elif m == m_list[0]:  # initialize and add 2 factor
        P = 2 * np.pi * ufl.inner(-ufl.cross(Esh_m,
                                  ufl.conj(Hsh_m)), n_3d) * marker
        Q = 2 * np.pi * eps_au.imag * k0 * (ufl.inner(Eh_m, Eh_m)) / Z0 / n_bkg

        q_abs_fenics_proc = (fem.assemble_scalar(
            fem.form(Q * rho * dAu)) / gcs / I0).real
        q_sca_fenics_proc = (fem.assemble_scalar(
                             fem.form((P('+') + P('-')) * rho * dS(scatt_tag))) / gcs / I0).real
        q_abs_fenics = domain.comm.allreduce(q_abs_fenics_proc, op=MPI.SUM)
        q_sca_fenics = domain.comm.allreduce(q_sca_fenics_proc, op=MPI.SUM)

    else:  # do not initialize and add 2 factor
        P = 2 * np.pi * ufl.inner(-ufl.cross(Esh_m,
                                  ufl.conj(Hsh_m)), n_3d) * marker
        Q = 2 * np.pi * eps_au.imag * k0 * (ufl.inner(Eh_m, Eh_m)) / Z0 / n_bkg

        q_abs_fenics_proc = (fem.assemble_scalar(
            fem.form(Q * rho * dAu)) / gcs / I0).real
        q_sca_fenics_proc = (fem.assemble_scalar(
                             fem.form((P('+') + P('-')) * rho * dS(scatt_tag))) / gcs / I0).real
        q_abs_fenics += domain.comm.allreduce(q_abs_fenics_proc, op=MPI.SUM)
        q_sca_fenics += domain.comm.allreduce(q_sca_fenics_proc, op=MPI.SUM)

q_ext_fenics = q_abs_fenics + q_sca_fenics
# -

# The quantities `P` and `Q` have an additional `2` factor for `m != 0` due to parity.

# We can now compare the numerical efficiencies with
# the analytical ones. The latter were obtained
# with the following routine
# provided by the [`scattnlay`](https://github.com/ovidiopr/scattnlay)
# library:
#
# ```
# from scattnlay import scattnlay
#
# m = np.sqrt(eps_au)/n_bkg
# x = 2*np.pi*radius_sph/wl0*n_bkg
#
# q_sca_analyt, q_abs_analyt = scattnlay(np.array([x], dtype=np.complex128), np.array([m], dtype=np.complex128))[2:4]
# ```
#
# The numerical values are reported here below:

q_abs_analyt = 0.9622728008329892
q_sca_analyt = 0.07770397394691526
q_ext_analyt = q_abs_analyt + q_sca_analyt

# Finally, we can calculate the error and save our total
# scattered field:

# +
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

    # Check whether the geometrical and optical parameters ar correct
    assert radius_sph / wl0 == 0.025 / 0.4
    assert eps_au == -1.0782 + 1j * 5.8089

    assert err_abs < 0.01
    assert err_sca < 0.01
    assert err_ext < 0.01

if has_vtx:
    v_dg_el = ufl.VectorElement("DG", domain.ufl_cell(), degree, dim=3)
    W = fem.FunctionSpace(domain, v_dg_el)
    Es_dg = fem.Function(W)
    Es_expr = fem.Expression(Esh, W.element.interpolation_points())
    Es_dg.interpolate(Es_expr)

    with VTXWriter(domain.comm, "sols/Es.bp", Es_dg) as f:
        f.write(0.0)
