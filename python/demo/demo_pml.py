# # Electromagnetic scattering from a wire with PML
#
# Copyright (C) 2022-2025 Michele Castriotta, Igor Baratta
# and Jørgen S. Dokken
#
# ```{admonition} Download sources
# :class: download
# * {download}`Python script <./demo_pml.py>`
# * {download}`Jupyter notebook <./demo_pml.ipynb>`
# ```
#
# This demo illustrates how to:
# - Use complex quantities in FEniCSx
# - Setup and solve Maxwell's equations
# - Implement (rectangular) perfectly matched layers (PMLs)
# - Use custom integration entities for one-sided interior facet integrals
#
# First, we import the required modules

# +
import sys
from functools import partial, reduce
from pathlib import Path

from mpi4py import MPI
from petsc4py import PETSc

import gmsh
import numpy as np
from scipy.special import h2vp, hankel2, jv, jvp

import dolfinx
import ufl
from basix.ufl import element
from dolfinx import default_real_type, default_scalar_type, fem, mesh, plot
from dolfinx.fem.petsc import LinearProblem
from dolfinx.io import gmsh as gmshio

try:
    from dolfinx.io import VTXWriter
except ImportError:
    print("This demo requires DOLFINx to be configured with adios2.")
    exit(0)


try:
    import pyvista

    have_pyvista = True
except ModuleNotFoundError:
    print("pyvista and pyvistaqt are required to visualise the solution")
    have_pyvista = False
# -

# Since we want to solve time-harmonic Maxwell's equation, we require
# that the demo is executed with DOLFINx (PETSc) complex mode.

if not np.issubdtype(default_scalar_type, np.complexfloating):
    print("Demo should only be executed with DOLFINx complex mode")
    exit(0)

# # Mesh generation with GMSH
# The mesh is made up by a central circle (the wire), and an external
# layer (the PML) divided in 4 rectangles and 4 squares at the corners.
# The `generate_mesh_wire` function takes as input:

# - `radius_wire`: the radius of the wire
# - `radius_scatt`: the radius of the circle where scattering efficiency
#   is calculated
# - `l_dom`: length of real domain
# - `l_pml`: length of PML layer
# - `in_wire_size`: the mesh size at a distance `0.8 * radius_wire` from
#   the origin
# - `on_wire_size`: the mesh size on the wire boundary
# - `scatt_size`: the mesh size on the circle where scattering
#   efficiency is calculated
# - `pml_size`: the mesh size on the outer boundary of the PML
# - `au_tag`: the tag of the physical group representing the wire
# - `bkg_tag`: the tag of the physical group representing the background
# - `scatt_tag`: the tag of the physical group representing the boundary
#   where scattering efficiency is calculated
# - `pml_tag`: the tag of the physical group representing the PML
#   (together with pml_tag+1 and pml_tag+2)
#
#


def generate_mesh_wire(
    radius_wire: float,
    radius_scatt: float,
    l_dom: float,
    l_pml: float,
    in_wire_size: float,
    on_wire_size: float,
    scatt_size: float,
    pml_size: float,
    au_tag: int,
    bkg_tag: int,
    scatt_tag: int,
    pml_tag: int,
):
    dim = 2
    # A dummy circle for setting a finer mesh
    c1 = gmsh.model.occ.addCircle(0.0, 0.0, 0.0, radius_wire * 0.8, angle1=0.0, angle2=2 * np.pi)
    gmsh.model.occ.addCurveLoop([c1], tag=c1)
    gmsh.model.occ.addPlaneSurface([c1], tag=c1)

    c2 = gmsh.model.occ.addCircle(0.0, 0.0, 0.0, radius_wire, angle1=0, angle2=2 * np.pi)
    gmsh.model.occ.addCurveLoop([c2], tag=c2)
    gmsh.model.occ.addPlaneSurface([c2], tag=c2)
    wire, _ = gmsh.model.occ.fragment([(dim, c2)], [(dim, c1)])

    # A dummy circle for the calculation of the scattering efficiency
    c3 = gmsh.model.occ.addCircle(0.0, 0.0, 0.0, radius_scatt, angle1=0, angle2=2 * np.pi)
    gmsh.model.occ.addCurveLoop([c3], tag=c3)
    gmsh.model.occ.addPlaneSurface([c3], tag=c3)

    r0 = gmsh.model.occ.addRectangle(-l_dom / 2, -l_dom / 2, 0, l_dom, l_dom)
    inclusive_rectangle, _ = gmsh.model.occ.fragment([(dim, r0)], [(dim, c3)])

    delta_pml = (l_pml - l_dom) / 2

    separate_rectangle, _ = gmsh.model.occ.cut(inclusive_rectangle, wire, removeTool=False)
    _, physical_domain = gmsh.model.occ.fragment(separate_rectangle, wire)

    bkg_tags = [tag[0] for tag in physical_domain[: len(separate_rectangle)]]

    wire_tags = [
        tag[0]
        for tag in physical_domain[len(separate_rectangle) : len(inclusive_rectangle) + len(wire)]
    ]

    # Corner PMLS
    pml1 = gmsh.model.occ.addRectangle(-l_pml / 2, l_dom / 2, 0, delta_pml, delta_pml)
    pml2 = gmsh.model.occ.addRectangle(-l_pml / 2, -l_pml / 2, 0, delta_pml, delta_pml)
    pml3 = gmsh.model.occ.addRectangle(l_dom / 2, l_dom / 2, 0, delta_pml, delta_pml)
    pml4 = gmsh.model.occ.addRectangle(l_dom / 2, -l_pml / 2, 0, delta_pml, delta_pml)
    corner_pmls = [(dim, pml1), (dim, pml2), (dim, pml3), (dim, pml4)]

    # X pmls
    pml5 = gmsh.model.occ.addRectangle(-l_pml / 2, -l_dom / 2, 0, delta_pml, l_dom)
    pml6 = gmsh.model.occ.addRectangle(l_dom / 2, -l_dom / 2, 0, delta_pml, l_dom)
    x_pmls = [(dim, pml5), (dim, pml6)]

    # Y pmls
    pml7 = gmsh.model.occ.addRectangle(-l_dom / 2, l_dom / 2, 0, l_dom, delta_pml)
    pml8 = gmsh.model.occ.addRectangle(-l_dom / 2, -l_pml / 2, 0, l_dom, delta_pml)
    y_pmls = [(dim, pml7), (dim, pml8)]
    _, surface_map = gmsh.model.occ.fragment(bkg_tags + wire_tags, corner_pmls + x_pmls + y_pmls)

    gmsh.model.occ.synchronize()

    bkg_group = [tag[0][1] for tag in surface_map[: len(bkg_tags)]]
    gmsh.model.addPhysicalGroup(dim, bkg_group, tag=bkg_tag)
    wire_group = [tag[0][1] for tag in surface_map[len(bkg_tags) : len(bkg_tags + wire_tags)]]

    gmsh.model.addPhysicalGroup(dim, wire_group, tag=au_tag)

    corner_group = [
        tag[0][1]
        for tag in surface_map[len(bkg_tags + wire_tags) : len(bkg_tags + wire_tags + corner_pmls)]
    ]
    gmsh.model.addPhysicalGroup(dim, corner_group, tag=pml_tag)

    x_group = [
        tag[0][1]
        for tag in surface_map[
            len(bkg_tags + wire_tags + corner_pmls) : len(
                bkg_tags + wire_tags + corner_pmls + x_pmls
            )
        ]
    ]

    gmsh.model.addPhysicalGroup(dim, x_group, tag=pml_tag + 1)

    y_group = [
        tag[0][1]
        for tag in surface_map[
            len(bkg_tags + wire_tags + corner_pmls + x_pmls) : len(
                bkg_tags + wire_tags + corner_pmls + x_pmls + y_pmls
            )
        ]
    ]

    gmsh.model.addPhysicalGroup(dim, y_group, tag=pml_tag + 2)

    # Marker interior surface in bkg group
    boundaries: list[np.typing.NDArray[np.int32]] = []
    for tag in bkg_group:
        boundary_pairs = gmsh.model.get_boundary([(dim, tag)], oriented=False)
        boundaries.append(np.asarray([pair[1] for pair in boundary_pairs], dtype=np.int32))

    interior_boundary = reduce(np.intersect1d, boundaries)
    gmsh.model.addPhysicalGroup(dim - 1, interior_boundary, tag=scatt_tag)
    gmsh.model.mesh.setSize([(0, 1)], size=in_wire_size)
    gmsh.model.mesh.setSize([(0, 2)], size=on_wire_size)
    gmsh.model.mesh.setSize([(0, 3)], size=scatt_size)
    gmsh.model.mesh.setSize([(0, x) for x in range(4, 40)], size=pml_size)

    gmsh.model.mesh.generate(2)
    return gmsh.model


# ## Mathematical formulation
# Following are convenience functions for the calculation of the
# absorption, scattering and extinction efficiencies of a wire
# being hit normally by a TM-polarized electromagnetic wave.
# See {ref}`Scattering boundary conditions:
# Mathematical formulation <em_efficiencies>` for a detailed description.

# +


def compute_a(nu: int, m: complex, alpha: float) -> float:
    J_nu_alpha = jv(nu, alpha)
    J_nu_malpha = jv(nu, m * alpha)
    J_nu_alpha_p = jvp(nu, alpha, 1)
    J_nu_malpha_p = jvp(nu, m * alpha, 1)

    H_nu_alpha = hankel2(nu, alpha)
    H_nu_alpha_p = h2vp(nu, alpha, 1)

    a_nu_num = J_nu_alpha * J_nu_malpha_p - m * J_nu_malpha * J_nu_alpha_p
    a_nu_den = H_nu_alpha * J_nu_malpha_p - m * J_nu_malpha * H_nu_alpha_p
    return a_nu_num / a_nu_den


def calculate_analytical_efficiencies(
    eps: complex, n_bkg: float, wl0: float, radius_wire: float, num_n: int = 50
) -> tuple[float, float, float]:
    m = np.sqrt(np.conj(eps)) / n_bkg
    alpha = 2 * np.pi * radius_wire / wl0 * n_bkg
    c = 2 / alpha
    q_ext = c * np.real(compute_a(0, m, alpha))
    q_sca = c * np.abs(compute_a(0, m, alpha)) ** 2
    for nu in range(1, num_n + 1):
        q_ext += c * 2 * np.real(compute_a(nu, m, alpha))
        q_sca += c * 2 * np.abs(compute_a(nu, m, alpha)) ** 2
    return q_ext - q_sca, q_sca, q_ext


# -

# ## Perfectly matched layers (PMLs)
# Now, let's consider an infinite metallic wire immersed in a background
# medium (e.g. vacuum or water). Let's now consider the plane cutting
# the wire perpendicularly to its axis at a generic point. Such plane
# $\Omega=\Omega_{m} \cup\Omega_{b}$ is formed by the cross-section of
# the wire $\Omega_m$ and the background medium $\Omega_{b}$ surrounding
# the wire. We limit the background medium with a squared perfectly
# matched layer (or shortly PML), which will act as an absorber for
# outgoing scattered waves.
#
# The goal of this demo is to calculate the electric field
# $\mathbf{E}_s$ scattered by the wire when a background wave
# $\mathbf{E}_b$ impinges on it. We will consider a background plane
# wave at $\lambda_0$ wavelength, which can be written analytically as:
#
# $$
# \mathbf{E}_b = \exp(\mathbf{k}\cdot\mathbf{r})\hat{\mathbf{u}}_p
# $$
#
# with $\mathbf{k} = \frac{2\pi}{\lambda_0}n_b\hat{\mathbf{u}}_k$ being
# the wavevector of the plane wave, pointing along the propagation
# direction, with $\hat{\mathbf{u}}_p$ being the polarization direction,
# and with $\mathbf{r}$ being a point in $\Omega$. We will only consider
# $\hat{\mathbf{u}}_k$ and $\hat{\mathbf{u}}_p$ with components
# belonging to the $\Omega$ domain and perpendicular to each other, i.e.
# $\hat{\mathbf{u}}_k \perp \hat{\mathbf{u}}_p$ (transversality
# condition of plane waves). Using a Cartesian coordinate system for
# $\Omega$, and by defining $k_x = n_bk_0\cos\theta$ and $k_y =
# n_bk_0\sin\theta$, with $\theta$ being the angle defined by the
# propagation direction $\hat{\mathbf{u}}_k$ and the horizontal axis
# $\hat{\mathbf{u}}_x$, we have:
#
# $$
# \mathbf{E}_b = -\sin\theta e^{j (k_xx+k_yy)}\hat{\mathbf{u}}_x
# + \cos\theta e^{j (k_xx+k_yy)}\hat{\mathbf{u}}_y
# $$
#
# The function `background_field` below implements this analytical
# formula:


# +
def background_field(theta: float, n_b: float, k0: complex, x: np.typing.NDArray[np.float64]):
    kx = n_b * k0 * np.cos(theta)
    ky = n_b * k0 * np.sin(theta)
    phi = kx * x[0] + ky * x[1]
    return (-np.sin(theta) * np.exp(1j * phi), np.cos(theta) * np.exp(1j * phi))


# -

# For convenience, we define the $\nabla\times$ operator for a 2D vector


def curl_2d(a: fem.Function):
    return ufl.as_vector((0, 0, a[1].dx(0) - a[0].dx(1)))


# Let's now see how we can implement PMLs for our problem. PMLs are
# artificial layers surrounding the real domain that gradually absorb
# waves impinging them. Mathematically, we can use a complex coordinate
# transformation of this kind to obtain this absorption:
#
# $$
# x^\prime= x\left\{1+j\frac{\alpha}{k_0}\left[\frac{|x|-l_{dom}/2}
# {(l_{pml}/2 - l_{dom}/2)^2}\right] \right\}
# $$
#
# with $l_{dom}$ and $l_{pml}$ being the lengths of the domain without
# and with PML, respectively, and with $\alpha$ being a parameter that
# tunes the absorption within the PML (the bigger the $\alpha$, the
# faster the absorption). In DOLFINx, we can define this coordinate
# transformation in the following way:


def pml_coordinates(x: ufl.indexed.Indexed, alpha: float, k0: complex, l_dom: float, l_pml: float):
    return x + 1j * alpha / k0 * x * (ufl.algebra.Abs(x) - l_dom / 2) / (l_pml / 2 - l_dom / 2) ** 2


# We use the following domain specific parameters:

# +
epsilon_0 = 8.8541878128 * 10**-12
mu_0 = 4 * np.pi * 10**-7

# Radius of the wire and of the boundary of the domain
radius_wire = 0.05
l_dom = 0.8
radius_scatt = 0.8 * l_dom / 2
l_pml = 1

mesh_factor = 1  # The smaller the mesh_factor, the finer is the mesh
in_wire_size = mesh_factor * 6e-3  # Mesh size inside the wire
on_wire_size = mesh_factor * 3.0e-3  # Mesh size at the boundary of the wire
scatt_size = mesh_factor * 15.0e-3  # Mesh size in the background
pml_size = mesh_factor * 15.0e-3  # Mesh size at the boundary


# Tags for the subdomains
au_tag = 1
bkg_tag = 2
scatt_tag = 3
pml_tag = 4
# -

# We generate the mesh using GMSH and convert it to a
# {py:class}`Mesh<dolfinx.mesh.Mesh>` using
# {py:func}`model_to_mesh <dolfinx.io.gmsh.model_to_mesh>`.

# +
model = None
gmsh.initialize(sys.argv)
if MPI.COMM_WORLD.rank == 0:
    model = generate_mesh_wire(
        radius_wire,
        radius_scatt,
        l_dom,
        l_pml,
        in_wire_size,
        on_wire_size,
        scatt_size,
        pml_size,
        au_tag,
        bkg_tag,
        scatt_tag,
        pml_tag,
    )
model = MPI.COMM_WORLD.bcast(model, root=0)
partitioner = dolfinx.cpp.mesh.create_cell_partitioner(dolfinx.mesh.GhostMode.shared_facet)

mesh_data = gmshio.model_to_mesh(model, MPI.COMM_WORLD, 0, gdim=2, partitioner=partitioner)
assert mesh_data.cell_tags is not None, "Cell tags are missing"
assert mesh_data.facet_tags is not None, "Facet tags are missing"
assert all(pg.dim == 2 for _, pg in mesh_data.physical_groups.items()), "Wrong phsyical group dim."

gmsh.finalize()
MPI.COMM_WORLD.barrier()
# -

# We visualize the mesh and subdomains with
# [PyVista](https://docs.pyvista.org/)

out_folder = Path("output_pml")
out_folder.mkdir(parents=True, exist_ok=True)
tdim = mesh_data.mesh.topology.dim
if have_pyvista:
    topology, cell_types, geometry = plot.vtk_mesh(mesh_data.mesh, 2)
    grid = pyvista.UnstructuredGrid(topology, cell_types, geometry)
    plotter = pyvista.Plotter()
    num_local_cells = mesh_data.mesh.topology.index_map(tdim).size_local
    grid.cell_data["Marker"] = mesh_data.cell_tags.values[
        mesh_data.cell_tags.indices < num_local_cells
    ]
    grid.set_active_scalars("Marker")
    plotter.add_mesh(grid, show_edges=True)
    plotter.view_xy()
    if not pyvista.OFF_SCREEN:
        plotter.show(interactive=True)
    else:
        figure = plotter.screenshot(out_folder / "wire_mesh_pml.png", window_size=[800, 800])

# We observe five different subdomains: one for the gold wire
# (`au_tag`), one for the background medium (`bkg_tag`), one for the PML
# corners (`pml_tag`), one for the PML rectangles along $x$ (`pml_tag +
# 1`), and one for the PML rectangles along $y$ (`pml_tag + 2`). These
# different PML regions have different coordinate transformation, as
# specified here below:
#
# $$
# \begin{align}
# \text{PML}_\text{corners} \rightarrow \mathbf{r}^\prime &= (x^\prime,
#   y^\prime) \\
# \text{PML}_\text{rectangles along x} \rightarrow
#   \mathbf{r}^\prime &= (x^\prime, y) \\
# \text{PML}_\text{rectangles along y} \rightarrow
#   \mathbf{r}^\prime &= (x, y^\prime).
# \end{align}
# $$
#
# Now we define some other problem specific parameters:

wl0 = 0.4  # Wavelength of the background field
n_bkg = 1  # Background refractive index
eps_bkg = n_bkg**2  # Background relative permittivity
k0 = 2 * np.pi / wl0  # Wavevector of the background field
theta = 0  # Angle of incidence of the background field

# We use a degree 3
# [Nedelec (first kind)](https://defelement.org/elements/nedelec1.html)
# element to represent the electric field:

degree = 3
curl_el = element("N1curl", mesh_data.mesh.basix_cell(), degree, dtype=default_real_type)
V = fem.functionspace(mesh_data.mesh, curl_el)

# Next, we interpolate $\mathbf{E}_b$ into the function space $V$,
# define our trial and test function, and the integration domains:

# +
Eb = fem.Function(V)
f = partial(background_field, theta, n_bkg, k0)
Eb.interpolate(f)

# Definition of Trial and Test functions
Es = ufl.TrialFunction(V)
v = ufl.TestFunction(V)

# Definition of 3d fields
Es_3d = ufl.as_vector((Es[0], Es[1], 0))
v_3d = ufl.as_vector((v[0], v[1], 0))

# Measures for subdomains
dx = ufl.Measure("dx", mesh_data.mesh, subdomain_data=mesh_data.cell_tags)
dDom = dx((au_tag, bkg_tag))
dPml_xy = dx(pml_tag)
dPml_x = dx(pml_tag + 1)
dPml_y = dx(pml_tag + 2)
# -

# Let's now define the relative permittivity $\varepsilon_m$ of the gold
# wire at $400nm$ (data taken from [*Olmon et al.
# 2012*](https://doi.org/10.1103/PhysRevB.86.235147) , and for a quick
# reference have a look at [refractiveindex.info](
# https://refractiveindex.info/?shelf=main&book=Au&page=Olmon-sc)):

# Definition of relative permittivity for Au @400nm
eps_au = -1.0782 + 1j * 5.8089


# We can now define a space function for the permittivity $\varepsilon$
# that takes the value $\varepsilon_m$ for cells inside the wire, while
# it takes the value of the background permittivity $\varepsilon_b$ in
# the background region:

D = fem.functionspace(mesh_data.mesh, ("DG", 0))
eps = fem.Function(D)
au_cells = mesh_data.cell_tags.find(au_tag)
bkg_cells = mesh_data.cell_tags.find(bkg_tag)
eps.x.array[au_cells] = np.full_like(au_cells, eps_au, dtype=eps.x.array.dtype)
eps.x.array[bkg_cells] = np.full_like(bkg_cells, eps_bkg, dtype=eps.x.array.dtype)
eps.x.scatter_forward()

# Now we need to define our weak form in DOLFINx. Let's write the PML
# weak form first. As a first step, we can define our new complex
# coordinates as:

# +
x = ufl.SpatialCoordinate(mesh_data.mesh)
alpha = 1

# PML corners
xy_pml = ufl.as_vector(
    (pml_coordinates(x[0], alpha, k0, l_dom, l_pml), pml_coordinates(x[1], alpha, k0, l_dom, l_pml))
)

# PML rectangles along x
x_pml = ufl.as_vector((pml_coordinates(x[0], alpha, k0, l_dom, l_pml), x[1]))

# PML rectangles along y
y_pml = ufl.as_vector((x[0], pml_coordinates(x[1], alpha, k0, l_dom, l_pml)))
# -

# We can then express this coordinate systems as a material
# transformation within the PML region. In other words, the PML region
# can be interpreted as a material having, in general, anisotropic,
# inhomogeneous and complex permittivity
# $\boldsymbol{\varepsilon}_{pml}$ and permeability
# $\boldsymbol{\mu}_{pml}$. To do this, we need to calculate the
# Jacobian of the coordinate transformation:
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
# $\boldsymbol{\mu}_{pml}$ can be calculated with the following formula,
# from [Ward & Pendry, 1996](
# https://www.tandfonline.com/doi/abs/10.1080/09500349608232782):
#
# $$
# \begin{align}
# {\boldsymbol{\varepsilon}_{pml}} &=
# A^{-1} \mathbf{A} {\boldsymbol{\varepsilon}_b}\mathbf{A}^{T},\\
# {\boldsymbol{\mu}_{pml}} &=
# A^{-1} \mathbf{A} {\boldsymbol{\mu}_b}\mathbf{A}^{T},
# \end{align}
# $$
#
# with $A^{-1}=\operatorname{det}(\mathbf{J})$.
#
# We use {py:func}`ufl.grad` to calculate the Jacobian of our coordinate
# transformation for the different PML regions, and then we can
# implement this Jacobian for calculating
# $\boldsymbol{\varepsilon}_{pml}$ and $\boldsymbol{\mu}_{pml}$. The
# here below function named `create_eps_mu()` serves this purpose:

# +


def create_eps_mu(
    pml: ufl.tensors.ListTensor,
    eps_bkg: float | ufl.tensors.ListTensor,
    mu_bkg: float | ufl.tensors.ListTensor,
) -> tuple[ufl.tensors.ComponentTensor, ufl.tensors.ComponentTensor]:
    J = ufl.grad(pml)

    # Transform the 2x2 Jacobian into a 3x3 matrix.
    J = ufl.as_matrix(((J[0, 0], 0, 0), (0, J[1, 1], 0), (0, 0, 1)))

    A = ufl.inv(J)
    eps_pml = ufl.det(J) * A * eps_bkg * ufl.transpose(A)
    mu_pml = ufl.det(J) * A * mu_bkg * ufl.transpose(A)
    return eps_pml, mu_pml


eps_x, mu_x = create_eps_mu(x_pml, eps_bkg, 1)
eps_y, mu_y = create_eps_mu(y_pml, eps_bkg, 1)
eps_xy, mu_xy = create_eps_mu(xy_pml, eps_bkg, 1)

# -

# The final weak form in the PML region is:
#
# $$
# \int_{\Omega_{pml}}\left[\boldsymbol{\mu}^{-1}_{pml} \nabla \times
# \mathbf{E} \right]\cdot \nabla \times \bar{\mathbf{v}}-k_{0}^{2}
# \left[\boldsymbol{\varepsilon}_{pml} \mathbf{E} \right]\cdot
# \bar{\mathbf{v}}~ d x=0,
# $$
#
#
# while in the rest of the domain is:
#
# $$
# \int_{\Omega_m\cup\Omega_b}-(\nabla \times \mathbf{E}_s)
# \cdot (\nabla \times \bar{\mathbf{v}})+\varepsilon_{r} k_{0}^{2}
# \mathbf{E}_s \cdot \bar{\mathbf{v}}+k_{0}^{2}\left(\varepsilon_{r}
# -\varepsilon_b\right)\mathbf{E}_b \cdot \bar{\mathbf{v}}~\mathrm{d}x.
# = 0.
# $$
#
# Let's solve this equation in DOLFINx:

# +
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

a, L = ufl.lhs(F), ufl.rhs(F)

# For factorisation prefer MUMPS, then superlu_dist, then default
sys = PETSc.Sys()  # type: ignore
use_superlu = PETSc.IntType == np.int64
if sys.hasExternalPackage("mumps") and not use_superlu:  # type: ignore
    mat_factor_backend = "mumps"
elif sys.hasExternalPackage("superlu_dist"):  # type: ignore
    mat_factor_backend = "superlu_dist"
else:
    if mesh_data.mesh.comm > 1:
        raise RuntimeError("This demo requires a parallel LU solver.")
    else:
        mat_factor_backend = "petsc"

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
# -

# Let's now save the solution in a `bp`-file. In order to do so, we need
# to interpolate our solution discretized with Nedelec elements into a
# compatible discontinuous Lagrange space.

# +
gdim = mesh_data.mesh.geometry.dim
V_dg = fem.functionspace(mesh_data.mesh, ("DG", degree, (gdim,)))
Esh_dg = fem.Function(V_dg)
Esh_dg.interpolate(Esh)

with VTXWriter(mesh_data.mesh.comm, out_folder / "Esh.bp", Esh_dg) as vtx:
    vtx.write(0.0)
# -

# For more information about saving and visualizing vector fields
# discretized with Nedelec elements, check [this](./demo_interpolation-io)
# DOLFINx demo.

if have_pyvista:
    V_cells, V_types, V_x = plot.vtk_mesh(V_dg)
    V_grid = pyvista.UnstructuredGrid(V_cells, V_types, V_x)
    Esh_values = np.zeros((V_x.shape[0], 3), dtype=np.float64)
    Esh_values[:, :tdim] = Esh_dg.x.array.reshape(V_x.shape[0], tdim).real
    V_grid.point_data["u"] = Esh_values

    plotter = pyvista.Plotter()
    plotter.add_text("magnitude", font_size=12, color="black")
    plotter.add_mesh(V_grid.copy(), show_edges=True)
    plotter.view_xy()
    plotter.link_views()

    if not pyvista.OFF_SCREEN:
        plotter.show()
    else:
        plotter.screenshot(out_folder / "Esh.png", window_size=[800, 800])

# Next we can calculate the total electric field
# $\mathbf{E}=\mathbf{E}_s+\mathbf{E}_b$ and save it:

# +
E = fem.Function(V)
E.x.array[:] = Eb.x.array[:] + Esh.x.array[:]

E_dg = fem.Function(V_dg)
E_dg.interpolate(E)

with VTXWriter(mesh_data.mesh.comm, out_folder / "E.bp", E_dg) as vtx:
    vtx.write(0.0)
# -

# ## Post-processing
#
# To validate the formulation we calculate the absorption, scattering
# and extinction efficiencies, which are quantities that define how much
# light is absorbed and scattered by the wire. First of all, we
# calculate the analytical efficiencies with the
# `calculate_analytical_efficiencies` function

q_abs_analyt, q_sca_analyt, q_ext_analyt = calculate_analytical_efficiencies(
    eps_au, n_bkg, wl0, radius_wire
)

# We calculate the numerical efficiencies in the same way as done in
# [Electromagnetic scattering demo](./demo_scattering_boundary_conditions),
# with the only difference that now the scattering efficiency needs to be
# calculated over an inner facet, and therefore it requires a slightly
# different approach:

# ### One-sided interior facet integrals

# An integration entity of an integral over a single facet in DOLFINx is
# defined as a tuple, `(cell_idx, local_facet_idx)`, where `cell_idx` is
# the index of a cell containing the facet, (local to process),
# while `local_facet_idx` is the relative index of the facet in the cell.
# For exterior facets, this is straightforward to obtain,
# as a facet is then connected to only one cell.
# However, for an interior facet a facet is connected to at lest two cells.
# As we would like to compute the outwards facing flux through the wire,
# we want to be able to integrate only from the side of the cell that has
# a normal pointing outwards.
# We start by identifying all cells that are interior to the scatter tag.

cell_map = mesh_data.mesh.topology.index_map(tdim)
num_cells_local = cell_map.size_local + cell_map.num_ghosts
midpoints = mesh.compute_midpoints(mesh_data.mesh, tdim, np.arange(num_cells_local, dtype=np.int32))
is_inner_cell = (midpoints[:, 0] ** 2 + midpoints[:, 1] ** 2) < (radius_scatt) ** 2

# Next, we compute the integration entity for the facets in question.
# We start by finding all facets owned by the current process (to assure
# that we only integrate over each facet once), and then for each facet,
# we find the connected cells.

# +
# Get connectivity between facets and cells
mesh_data.mesh.topology.create_connectivity(tdim - 1, tdim)
mesh_data.mesh.topology.create_connectivity(tdim, tdim - 1)
f_to_c = mesh_data.mesh.topology.connectivity(tdim - 1, tdim)
c_to_f = mesh_data.mesh.topology.connectivity(tdim, tdim - 1)

# Filter facets to only those owned by the current process
num_facets_local = mesh_data.mesh.topology.index_map(tdim - 1).size_local
scatt_facets = mesh_data.facet_tags.find(scatt_tag)
scatt_facets = scatt_facets[scatt_facets < num_facets_local]

# Pack integration data
integration_entities = np.empty((len(scatt_facets), 2), dtype=np.int32)
for i, facet in enumerate(scatt_facets):
    connected_cells = f_to_c.links(facet)
    inner_cell_idx = np.flatnonzero(is_inner_cell[connected_cells])
    assert len(inner_cell_idx) == 1, "Expected exactly one inner cell per facet."
    inner_cell = connected_cells[inner_cell_idx[0]]
    local_facets = c_to_f.links(inner_cell)
    local_facet_idx = np.flatnonzero(local_facets == facet)[0]
    integration_entities[i, :] = (inner_cell, local_facet_idx)
# -

# Now that we have computed the integration entities, we define
# an integration measure for one-sided facet integrals `ufl.ds`:

ds_scatter = ufl.ds(
    domain=mesh_data.mesh,
    subdomain_data=[(scatt_tag, integration_entities.flatten())],
    subdomain_id=scatt_tag,
)


scatt_facets = mesh_data.facet_tags.find(scatt_tag)
incident_cells = mesh.compute_incident_entities(
    mesh_data.mesh.topology, scatt_facets, tdim - 1, tdim
)

# +
Z0 = np.sqrt(mu_0 / epsilon_0)  # Vacuum impedance
Hsh_3d = -1j * curl_2d(Esh) / (Z0 * k0 * n_bkg)  # Magnetic field H
Esh_3d = ufl.as_vector((Esh[0], Esh[1], 0))
E_3d = ufl.as_vector((E[0], E[1], 0))

# Intensity of the electromagnetic fields I0 = 0.5*E0**2/Z0
# E0 = np.sqrt(ax**2 + ay**2) = 1, see background_electric_field
I0 = 0.5 / Z0
gcs = 2 * radius_wire  # Geometrical cross section of the wire
n = ufl.FacetNormal(mesh_data.mesh)
n_3d = ufl.as_vector((n[0], n[1], 0))

# Quantities for the calculation of efficiencies
P = 0.5 * ufl.inner(ufl.cross(Esh_3d, ufl.conj(Hsh_3d)), n_3d)
Q = 0.5 * eps_au.imag * k0 * (ufl.inner(E_3d, E_3d)) / (Z0 * n_bkg)

# Normalized absorption efficiency
dAu = dx(au_tag)  # Define integration domain for the wire
q_abs_fenics_proc = (fem.assemble_scalar(fem.form(Q * dAu)) / (gcs * I0)).real
# Sum results from all MPI processes
q_abs_fenics = mesh_data.mesh.comm.allreduce(q_abs_fenics_proc, op=MPI.SUM)

# Normalized scattering efficiency
q_sca_fenics_proc = (fem.assemble_scalar(fem.form(P * ds_scatter)) / (gcs * I0)).real

# Sum results from all MPI processes
q_sca_fenics = mesh_data.mesh.comm.allreduce(q_sca_fenics_proc, op=MPI.SUM)

# Extinction efficiency
q_ext_fenics = q_abs_fenics + q_sca_fenics

# Error calculation
err_abs = np.abs(q_abs_analyt - q_abs_fenics) / q_abs_analyt
err_sca = np.abs(q_sca_analyt - q_sca_fenics) / q_sca_analyt
err_ext = np.abs(q_ext_analyt - q_ext_fenics) / q_ext_analyt

PETSc.Sys.Print(f"The analytical absorption efficiency is {q_abs_analyt}")
PETSc.Sys.Print(f"The numerical absorption efficiency is {q_abs_fenics}")
PETSc.Sys.Print(f"The error is {err_abs * 100}%")
PETSc.Sys.Print()
PETSc.Sys.Print(f"The analytical scattering efficiency is {q_sca_analyt}")
PETSc.Sys.Print(f"The numerical scattering efficiency is {q_sca_fenics}")
PETSc.Sys.Print(f"The error is {err_sca * 100}%")
PETSc.Sys.Print()
PETSc.Sys.Print(f"The analytical extinction efficiency is {q_ext_analyt}")
PETSc.Sys.Print(f"The numerical extinction efficiency is {q_ext_fenics}")
PETSc.Sys.Print(f"The error is {err_ext * 100}%")
# -

# Check if errors are smaller than 1%
assert err_abs < 0.01, "Error in absorption efficiency is too large"
# assert err_sca < 0.01
assert err_ext < 0.01, "Error in extinction efficiency is too large"
