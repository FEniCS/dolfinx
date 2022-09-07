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
# First, we import the required modules

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

from analytical_efficiencies_wire import calculate_analytical_efficiencies
from mesh_wire_pml import generate_mesh_wire

from dolfinx import fem, mesh, plot
from dolfinx.io import VTXWriter, gmshio
from ufl import (FacetNormal, FiniteElement, Measure, SpatialCoordinate,
                 TestFunction, TrialFunction, algebra, as_matrix, as_vector,
                 conj, cross, det, grad, inner, inv, lhs, rhs, sqrt, transpose)

from mpi4py import MPI
from petsc4py import PETSc

# -

# Since we want to solve time-harmonic Maxwell's equation, we require that the demo is
# executed with DOLFINx (PETSc) complex mode.

if not np.issubdtype(PETSc.ScalarType, np.complexfloating):
    print("Demo should only be executed with DOLFINx complex mode")
    exit(0)

# We consider an infinite metallic wire immersed in
# a background medium (e.g. vacuum or water). Let's now
# consider the plane cutting the wire perpendicularly to
# its axis at a generic point. Such plane $\Omega=\Omega_{m}
# \cup\Omega_{b}\cup\Omega_{pml}$ is formed by the cross-section
# of the wire $\Omega_m$, the background medium
# $\Omega_{b}$ surrounding the wire, and a squared perfectly
# matched layer (PML) surrounding the domains. PMLs are
# reflectionless layers that gradually absorb waves impinging
# on them, therefore allowing us to truncate the domain size.
#
# The goal of this demo is to calculate the electric field $\mathbf{E}_s$
# scattered by the wire when a background wave $\mathbf{E}_b$
# impinges on it. We will consider a background plane wave at
# $\lambda_0$ wavelength, which can be written analytically as:
#
# $$
# \mathbf{E}_b = \exp(\mathbf{k}\cdot\mathbf{r})\hat{\mathbf{u}}_p
# $$
#
# with $\mathbf{k} = \frac{2\pi}{\lambda_0}n_b\hat{\mathbf{u}}_k$
# being the wavevector of the
# plane wave, pointing along the propagation direction,
# with $\hat{\mathbf{u}}_p$ being the
# polarization direction, and with $\mathbf{r}$ being a
# point in $\Omega$.
# We will only consider $\hat{\mathbf{u}}_k$ and $\hat{\mathbf{u}}_p$
# with components belonging
# to the $\Omega$ domain and perpendicular to each other,
# i.e. $\hat{\mathbf{u}}_k \perp \hat{\mathbf{u}}_p$
# (transversality condition of plane waves).
# Using a Cartesian coordinate system for $\Omega$,
# and by defining $k_x = n_bk_0\cos\theta$ and
# $k_y = n_bk_0\sin\theta$, with $\theta$ being the angle
# defined by the propagation direction $\hat{\mathbf{u}}_k$
# and the horizontal axis $\hat{\mathbf{u}}_x$,
# we have:
#
# $$
# \mathbf{E}_b = -\sin\theta e^{j (k_xx+k_yy)}\hat{\mathbf{u}}_x
# + \cos\theta e^{j (k_xx+k_yy)}\hat{\mathbf{u}}_y
# $$
#
# The function `background_field` below implements this analytical
# formula:

# +


def background_field(theta: float, n_b: float, k0: complex,
                     x: np.typing.NDArray[np.float64]):

    kx = n_b * k0 * np.cos(theta)
    ky = n_b * k0 * np.sin(theta)
    phi = kx * x[0] + ky * x[1]
    return (-np.sin(theta) * np.exp(1j * phi), np.cos(theta) * np.exp(1j * phi))

# -

# For convenience, we define the $\nabla\times$ operator for a 2D vector

# +


def curl_2d(a):
    return as_vector((0, 0, a[1].dx(0) - a[0].dx(1)))

# -

# a PML gradually absorb waves impinging them. Mathematically, this effect can be embedded by using a complex
# coordinate system of this kind:
#
# \begin{align}
# & x^\prime= x\left\{1+j\frac{\alpha}{k_0}\left[\frac{|x|-l_{dom}/2}
# {(l_{pml}/2 - l_{dom}/2)^2}\right] \right\}\\
# \end{align}
#
# with $l_{dom}$ and $l_{pml}$ being the lengths of the domain
# without and with PML, respectively, and with $\alpha$ being a parameter
# that tunes the absorption within the PML (the greater the $\alpha$,
# the faster the absorption). In DOLFINx, we can define this
# coordinate transformation in the following way:


def pml_coordinates(x, alpha: float, k0: complex,
                    l_dom: float, l_pml: float):
    return (x + 1j * alpha / k0 * x
            * (algebra.Abs(x) - l_dom / 2)
            / (l_pml / 2 - l_dom / 2)**2)


# We use the following domain specific parameters

# +
# Constants
um = 1  # micron
nm = um * 10**-3  # nanometer
epsilon_0 = 8.8541878128 * 10**-12
mu_0 = 4 * np.pi * 10**-7

# Radius of the wire and of the boundary of the domain
radius_wire = 0.05 * um
l_dom = 0.8 * um
l_scatt = 0.8 * l_dom / 2
l_pml = 1 * um

# The smaller the mesh_factor, the finer is the mesh
mesh_factor = 1

# Mesh size inside the wire
in_wire_size = mesh_factor * 6 * nm

# Mesh size at the boundary of the wire
on_wire_size = mesh_factor * 3 * nm

# Mesh size in the background
scatt_size = mesh_factor * 15 * nm

# Mesh size at the boundary
pml_size = mesh_factor * 15 * nm

# Tags for the subdomains
au_tag = 1
bkg_tag = 2
scatt_tag = 3
pml_tag = 4
# -

# We generate the mesh using GMSH and convert it to a
# `dolfinx.mesh.Mesh`.

# +
model = generate_mesh_wire(
    radius_wire, l_scatt, l_dom, l_pml,
    in_wire_size, on_wire_size, scatt_size, pml_size,
    au_tag, bkg_tag, scatt_tag, pml_tag)

domain, cell_tags, facet_tags = gmshio.model_to_mesh(
    model, MPI.COMM_WORLD, 0, gdim=2)

gmsh.finalize()
MPI.COMM_WORLD.barrier()
# -

# We visualize the mesh and subdomains with [PyVista](https://docs.pyvista.org/)

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
        plotter.show(interactive=True)
    else:
        pyvista.start_xvfb()
        figure = plotter.screenshot("wire_mesh_pml.png",
                                    window_size=[800, 800])

# We observe five different subdomains: one for the gold
# wire (`au_tag`), one for the background medium (`bkg_tag`), one for the
# PML corners (`pml_tag`), one for the PML rectangles along $x$
# (`pml_tag + 1`), and one for the PML rectangles along $y$ (`pml_tag + 2`).
# These different PML regions have different coordinate transformation,
# as specified here below:
#
# \begin{align}
# \text{PML}_\text{corners} \rightarrow \mathbf{r}^\prime & = (x^\prime, y^\prime) \\
# \text{PML}_\text{rectangles along x} \rightarrow
#                                       \mathbf{r}^\prime & = (x^\prime, y) \\
# \text{PML}_\text{rectangles along y} \rightarrow
#                                       \mathbf{r}^\prime & = (x, y^\prime).
# \end{align}
#
# Now we define some other problem specific parameters:

wl0 = 0.4 * um  # Wavelength of the background field
n_bkg = 1  # Background refractive index
eps_bkg = n_bkg**2  # Background relative permittivity
k0 = 2 * np.pi / wl0  # Wavevector of the background field
deg = np.pi / 180
theta = 0 * deg  # Angle of incidence of the background field

# We use a 3rd order
# [Nedelec (first kind)](https://defelement.com/elements/nedelec1.html)
# element to represent the electric field:
#

degree = 3
curl_el = FiniteElement("N1curl", domain.ufl_cell(), degree)
V = fem.FunctionSpace(domain, curl_el)

# Next, we interpolate $\mathbf{E}_b$ into the function space $V$, define our
# trial and test function, and the integration domains:

# +
Eb = fem.Function(V)
f = partial(background_field, theta, n_bkg, k0)
Eb.interpolate(f)

# Definition of Trial and Test functions
Es = TrialFunction(V)
v = TestFunction(V)

# Definition of 3d fields
Es_3d = as_vector((Es[0], Es[1], 0))
v_3d = as_vector((v[0], v[1], 0))

# Measures for subdomains
dx = Measure("dx", domain, subdomain_data=cell_tags)
dDom = dx((au_tag, bkg_tag))
dPml_xy = dx(pml_tag)
dPml_x = dx(pml_tag + 1)
dPml_y = dx(pml_tag + 2)
# -

# Let's now define the relative permittivity $\varepsilon_m$
# of the gold wire at $400nm$ (data taken from
# [*Olmon et al. 2012*](https://doi.org/10.1103/PhysRevB.86.235147)
# , and for a quick reference have a look at [refractiveindex.info](
# https://refractiveindex.info/?shelf=main&book=Au&page=Olmon-sc
# )):

# Definition of relative permittivity for Au @400nm
eps_au = -1.0782 + 1j * 5.8089


# We can now define a space function for the permittivity
# $\varepsilon$ that takes the value $\varepsilon_m$
# for cells inside the wire, while it takes the value of the
# background permittivity $\varepsilon_b$ in the background region:

D = fem.FunctionSpace(domain, ("DG", 0))
eps = fem.Function(D)
au_cells = cell_tags.find(au_tag)
bkg_cells = cell_tags.find(bkg_tag)
eps.x.array[au_cells] = np.full_like(
    au_cells, eps_au, dtype=np.complex128)
eps.x.array[bkg_cells] = np.full_like(bkg_cells, eps_bkg, dtype=np.complex128)
eps.x.scatter_forward()

# Now we need to define our weak form in DOLFINx.
# Let's write the PML weak form first. As a first step,
# we can define our new complex coordinates as:

# +
x = SpatialCoordinate(domain)
alpha = 1

# PML corners
xy_pml = as_vector((pml_coordinates(x[0], alpha, k0, l_dom, l_pml),
                    pml_coordinates(x[1], alpha, k0, l_dom, l_pml)))

# PML rectangles along x
x_pml = as_vector((pml_coordinates(x[0], alpha, k0, l_dom, l_pml), x[1]))

# PML rectangles along y
y_pml = as_vector((x[0], pml_coordinates(x[1], alpha, k0, l_dom, l_pml)))
# -

# We can then express this coordinate systems as
# a material transformation within the PML region. In other words,
# the PML region can be interpreted as a material
# having, in general, anisotropic, inhomogeneous and complex permittivity
# $\boldsymbol{\varepsilon}_{pml}$ and
# permeability $\boldsymbol{\mu}_{pml}$. To do this, we need
# to calculate the Jacobian of the coordinate transformation:
#
# $$
# \mathbf{J}=\mathbf{A}^{-1}= \nabla\boldsymbol{x}^
# \prime(\boldsymbol{x}) =
# \left[\begin{array}{ccc}
# \frac{\partial x^{\prime}}{\partial x} &
# \frac{\partial y^{\prime}}{\partial x} &
# \frac{\partial z^{\prime}}{\partial x} \\
# \frac{\partial x^{\prime}}{\partial y} &
# \frac{\partial y^{\prime}}{\partial y} &
# \frac{\partial z^{\prime}}{\partial y} \\
# \frac{\partial x^{\prime}}{\partial z} &
# \frac{\partial y^{\prime}}{\partial z} &
# \frac{\partial z^{\prime}}{\partial z}
# \end{array}\right]=\left[\begin{array}{ccc}
# \frac{\partial x^{\prime}}{\partial x} & 0 & 0 \\
# 0 & \frac{\partial y^{\prime}}{\partial y} & 0 \\
# 0 & 0 & \frac{\partial z^{\prime}}{\partial z}
# \end{array}\right]=\left[\begin{array}{ccc}
# J_{11} & 0 & 0 \\
# 0 & J_{22} & 0 \\
# 0 & 0 & 1
# \end{array}\right]
# $$
#
# Then, our $\boldsymbol{\varepsilon}_{pml}$ and
# $\boldsymbol{\mu}_{pml}$ can be calculated with
# the following formula, from
# [Ward & Pendry, 1996](
# https://www.tandfonline.com/doi/abs/10.1080/09500349608232782):
#
# $$
# \begin{align}
# & {\boldsymbol{\varepsilon}_{pml}} =
# A^{-1} \mathbf{A} {\boldsymbol{\varepsilon}_b}\mathbf{A}^{T},\\
# & {\boldsymbol{\mu}_{pml}} =
# A^{-1} \mathbf{A} {\boldsymbol{\mu}_b}\mathbf{A}^{T},
# \end{align}
# $$
#
# with $A^{-1}=\operatorname{det}(\mathbf{J})$.
#
# In DOLFINx, we use
# `ufl.grad` to calculate the Jacobian of our
# coordinate transformation for the different PML regions,
# and then we can implement this Jacobian for
# calculating $\boldsymbol{\varepsilon}_{pml}$
# and $\boldsymbol{\mu}_{pml}$. The here below function
# named `create_eps_mu()` serves this purpose:

# +


def create_eps_mu(pml, eps_bkg, mu_bkg):

    J = grad(pml)

    # Transform the 2x2 Jacobian into a 3x3 matrix.
    J = as_matrix(((J[0, 0], 0, 0),
                   (0, J[1, 1], 0),
                   (0, 0, 1)))

    A = inv(J)
    eps_pml = det(J) * A * eps_bkg * transpose(A)
    mu_pml = det(J) * A * mu_bkg * transpose(A)
    return eps_pml, mu_pml


eps_x, mu_x = create_eps_mu(x_pml, eps_bkg, 1)
eps_y, mu_y = create_eps_mu(y_pml, eps_bkg, 1)
eps_xy, mu_xy = create_eps_mu(xy_pml, eps_bkg, 1)

# -

# The final weak form in the PML region is:
#
# $$
# \int_{\Omega_{pml}}\left[\boldsymbol{\mu}^{-1}_{pml} \nabla \times \mathbf{E}
# \right]\cdot \nabla \times \bar{\mathbf{v}}-k_{0}^{2}
# \left[\boldsymbol{\varepsilon}_{pml} \mathbf{E} \right]\cdot
# \bar{\mathbf{v}}~ d x=0,
# $$
#
#
# while in the rest of the domain is:
#
# $$
# \begin{align}
# & \int_{\Omega_m\cup\Omega_b}-(\nabla \times \mathbf{E}_s)
# \cdot (\nabla \times \bar{\mathbf{v}})+\varepsilon_{r} k_{0}^{2}
# \mathbf{E}_s \cdot \bar{\mathbf{v}}+k_{0}^{2}\left(\varepsilon_{r}
# -\varepsilon_b\right)\mathbf{E}_b \cdot \bar{\mathbf{v}}~\mathrm{d}x.
# = 0.
# \end{align}
# $$
#
# Let's solve this equation in DOLFINx:

# +
# Definition of the weak form
F = - inner(curl_2d(Es), curl_2d(v)) * dDom \
    + eps * k0 ** 2 * inner(Es, v) * dDom \
    + k0 ** 2 * (eps - eps_bkg) * inner(Eb, v) * dDom \
    - inner(inv(mu_x) * curl_2d(Es), curl_2d(v)) * dPml_x \
    - inner(inv(mu_y) * curl_2d(Es), curl_2d(v)) * dPml_y \
    - inner(inv(mu_xy) * curl_2d(Es), curl_2d(v)) * dPml_xy \
    + k0 ** 2 * inner(eps_x * Es_3d, v_3d) * dPml_x \
    + k0 ** 2 * inner(eps_y * Es_3d, v_3d) * dPml_y \
    + k0 ** 2 * inner(eps_xy * Es_3d, v_3d) * dPml_xy

a, L = lhs(F), rhs(F)

problem = fem.petsc.LinearProblem(a, L, bcs=[], petsc_options={
                                  "ksp_type": "preonly", "pc_type": "lu"})
Esh = problem.solve()
# -

# Let's now save the solution in a `bp`-file. In order to do so,
# we need to interpolate our solution discretized with Nedelec elements
# into a compatible discontinuous Lagrange space.

# +
V_dg = fem.VectorFunctionSpace(domain, ("DG", degree))
Esh_dg = fem.Function(V_dg)
Esh_dg.interpolate(Esh)

with VTXWriter(domain.comm, "Esh.bp", Esh_dg) as vtx:
    vtx.write(0.0)
# -

# For more information about saving and visualizing vector fields
# discretized with Nedelec elements, check [this](
# https://docs.fenicsproject.org/dolfinx/main/python/demos/demo_interpolation-io.html)
# DOLFINx demo.

if have_pyvista:
    V_cells, V_types, V_x = plot.create_vtk_mesh(V_dg)
    V_grid = pyvista.UnstructuredGrid(V_cells, V_types, V_x)
    Esh_values = np.zeros((V_x.shape[0], 3), dtype=np.float64)
    Esh_values[:, :domain.topology.dim] = \
        Esh_dg.x.array.reshape(V_x.shape[0], domain.topology.dim).real

    V_grid.point_data["u"] = Esh_values

    pyvista.set_jupyter_backend("pythreejs")
    plotter = pyvista.Plotter()

    plotter.add_text("magnitude", font_size=12, color="black")
    plotter.add_mesh(V_grid.copy(), show_edges=True)
    plotter.view_xy()
    plotter.link_views()

    if not pyvista.OFF_SCREEN:
        plotter.show()
    else:
        pyvista.start_xvfb()
        plotter.screenshot("Esh.png", window_size=[800, 800])

# Next we can calculate the total electric field
# $\mathbf{E}=\mathbf{E}_s+\mathbf{E}_b$ and save it:

# +
E = fem.Function(V)
E.x.array[:] = Eb.x.array[:] + Esh.x.array[:]

E_dg = fem.Function(V_dg)
E_dg.interpolate(E)

with VTXWriter(domain.comm, "E.bp", E_dg) as vtx:
    vtx.write(0.0)
# -

# ## Post-processing
# We use `dolfinx.fem.Expression` to interpolate the norm of the electric field
# into a suitable function space.
#
# $$
# ||\mathbf{E}_s|| = \sqrt{\mathbf{E}_s\cdot\bar{\mathbf{E}}_s}
# $$

lagr_el = FiniteElement("CG", domain.ufl_cell(), 2)
V_normEsh = fem.FunctionSpace(domain, lagr_el)
norm_expr = fem.Expression(sqrt(inner(Esh, Esh)), V_normEsh.element.interpolation_points())
normEsh = fem.Function(V_normEsh)
normEsh.interpolate(norm_expr)

# To validate the formulation we calculate the
# absorption, scattering and extinction efficiencies, which are
# quantities that define how much light is absorbed and scattered
# by the wire. First of all, we calculate the analytical efficiencies
# with the `calculate_analytical_efficiencies` function defined in a
# separate file:

q_abs_analyt, q_sca_analyt, q_ext_analyt = calculate_analytical_efficiencies(
    eps_au, n_bkg, wl0, radius_wire)

# We calculate the numerical efficiencies in the same way as done in (cite the first demo)

# +
# Vacuum impedance
Z0 = np.sqrt(mu_0 / epsilon_0)

# Magnetic field H
Hsh_3d = -1j * curl_2d(Esh) / Z0 / k0 / n_bkg

Esh_3d = as_vector((Esh[0], Esh[1], 0))
E_3d = as_vector((E[0], E[1], 0))

# Intensity of the electromagnetic fields I0 = 0.5*E0**2/Z0
# E0 = np.sqrt(ax**2 + ay**2) = 1, see background_electric_field
I0 = 0.5 / Z0

# Geometrical cross section of the wire
gcs = 2 * radius_wire

n = FacetNormal(domain)
n_3d = as_vector((n[0], n[1], 0))

marker = fem.Function(D)
scatt_facets = facet_tags.find(scatt_tag)
incident_cells = mesh.compute_incident_entities(domain, scatt_facets,
                                                domain.topology.dim - 1,
                                                domain.topology.dim)

midpoints = mesh.compute_midpoints(domain, domain.topology.dim, incident_cells)
inner_cells = incident_cells[(midpoints[:, 0]**2
                              + midpoints[:, 1]**2) < (l_scatt)**2]

marker.x.array[inner_cells] = 1

# Quantities for the calculation of efficiencies
P = 0.5 * inner(cross(Esh_3d, conj(Hsh_3d)), n_3d) * marker
Q = 0.5 * eps_au.imag * k0 * (inner(E_3d, E_3d)) / Z0 / n_bkg

# Define integration domain for the wire
dAu = dx(au_tag)

# Define integration facet for the scattering efficiency
dS = Measure("dS", domain, subdomain_data=facet_tags)

# Normalized absorption efficiency
q_abs_fenics_proc = (fem.assemble_scalar(fem.form(Q * dAu)) / gcs / I0).real
# Sum results from all MPI processes
q_abs_fenics = domain.comm.allreduce(q_abs_fenics_proc, op=MPI.SUM)

# Normalized scattering efficiency
q_sca_fenics_proc = (fem.assemble_scalar(
    fem.form((P('+') + P('-')) * dS(scatt_tag))) / gcs / I0).real

# Sum results from all MPI processes
q_sca_fenics = domain.comm.allreduce(q_sca_fenics_proc, op=MPI.SUM)

# Extinction efficiency
q_ext_fenics = q_abs_fenics + q_sca_fenics

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
# -

# Check if errors are smaller than 1%
assert err_abs < 0.01
assert err_sca < 0.01
assert err_ext < 0.01
