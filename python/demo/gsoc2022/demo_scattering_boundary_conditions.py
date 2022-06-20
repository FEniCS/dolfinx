# # Scattering from a wire with scattering boundary conditions
# This demo is implemented in two files: one for the mesh
# generation with gmsh, and one for the variational forms
# and the solver. It illustrates how to:
#
# - Use complex quantities in FEniCSx
# - Setup and solve Maxwell's equations
# - Add scattering boundary conditions for transparent boundaries
# - Implement Scattering Boundary Conditions as absorbing boundaries
# - Calculate absorption, scattering and extinction efficiencies
#
# ## Equations and problem definition
# Let's consider an infinite metallic wire immersed in
# a background medium (e.g. vacuum or water). Let's now
# consider the plane cutting the wire perpendicularly to
# its axis at a generic point. Such plane $\Omega=\Omega_{m}
# \cup\Omega_{b}$ is formed by the cross-section
# of the wire $\Omega_m$ and the background medium
# $\Omega_{b}$ surrounding the wire. Let's consider
# just the portion of this plane delimited by an external
# circular boundary $\partial \Omega$. We want to calculate
# the electric field $\mathbf{E}_s$ scattered by the wire
# when a background wave $\mathbf{E}_b$ impinges on it.
# We will consider a background plane wave at $\lambda_0$
# wavelength, that can be written analytically as:
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
# If we call $x$ and $y$ the horizontal
# and vertical axis in our $\Omega$ domain,
# and by defining $k_x = n_bk_0\cos\theta$ and
# $k_y = n_bk_0\sin\theta$, with $\theta$ being the angle
# defined by the propagation direction $\hat{\mathbf{u}}_k$
# and the horizontal axis $\hat{\mathbf{u}}_x$,
# we can write more explicitly:
#
# $$
# \mathbf{E}_b = -\sin\theta e^{j (k_xx+k_yy)}\hat{\mathbf{u}}_x
# + \cos\theta e^{j (k_xx+k_yy)}\hat{\mathbf{u}}_y
# $$
#
# The Maxwell's equation for scattering problems takes the following form:
#
# $$
# -\nabla \times \nabla \times \mathbf{E}_s+\varepsilon_{r} k_{0}^{2} \mathbf{E}_s
# +k_{0}^{2}\left(\varepsilon_{r}-\varepsilon_{b}\right)
# \mathbf{E}_{\mathrm{b}}=0 \textrm{ in } \Omega,
# $$
#
# where $k_0 = 2\pi/\lambda_0$ is the vacuum wavevector of the background
# field, $\varepsilon_b$ is the background relative permittivity and
# $\varepsilon_r$ is the relative permittivity as a function of space,
# i.e.:
#
# $$
# \varepsilon_r = \begin{cases}
# \varepsilon_m & \textrm{on }\Omega_m \\
# \varepsilon_b & \textrm{on }\Omega_b
# \end{cases}
# $$
#
# with $\varepsilon_m$ being the relative permittivity of the metallic
# wire. As reference values, we will consider $\lambda_0 = 400\textrm{nm}$
# (violet light), $\varepsilon_b = 1.33^2$ (relative permittivity of water),
# and $\varepsilon_m = -1.0782 + 5.8089\textrm{j}$ (relative permittivity of
# gold at $400\textrm{nm}$ (ref)).
#
# To make the system determined, we need to add some boundary conditions
# on $\partial \Omega$. A common approach is the use of scattering
# boundary conditions (ref), which make the boundary transparent for
# $\mathbf{E}_s$, allowing us to restric the computational boundary
# to a finite $\Omega$ domain. The first-order boundary conditions
# in the 2D case take the following form:
#
# $$\mathbf{n} \times
# \nabla \times \mathbf{E}_s+\left(j k_{0}n_b + \frac{1}{2r}
# \right) \mathbf{n} \times \mathbf{E}_s
# \times \mathbf{n}=0\quad \textrm{ on } \partial \Omega,
# $$

#
# with $n_b = \sqrt{\varepsilon_b}$ being the background refractive
# index, $\mathbf{n}$ being the normal vector to $\partial \Omega$,
# and $r = \sqrt{(x-x_s)^2 + (y-y_s)^2}$ being the distance of the
# $(x, y)$ point on $\partial\Omega$ from the wire centered in
# $(x_s, y_s)$. In our case we will consider
# the wire centered in the origin of our mesh, and therefore $r =
# \sqrt{x^2 + y^2}$.
#
# Now we need to find the weak form of the problem. First of all,
# we need to take the inner products of the equations with a
# complex test function $\mathbf{v}$, and then we need to integrate
# the terms over the corresponding domains:
#
# $$
# \begin{align}
# & \int_{\Omega}-\nabla \times( \nabla \times \mathbf{E}_s) \cdot
# \bar{\mathbf{v}}+\varepsilon_{r} k_{0}^{2} \mathbf{E}_s \cdot
# \bar{\mathbf{v}}+k_{0}^{2}\left(\varepsilon_{r}-\varepsilon_b\right)
# \mathbf{E}_b \cdot \bar{\mathbf{v}}~\mathrm{d}x \\ +& \int_{\partial \Omega}
# (\mathbf{n} \times \nabla \times \mathbf{E}_s) \cdot \bar{\mathbf{v}}
# +\left(j n_bk_{0}+\frac{1}{2r}\right) (\mathbf{n} \times \mathbf{E}_s
# \times \mathbf{n}) \cdot \bar{\mathbf{v}}~\mathrm{d}s=0
# \end{align}.
# $$
#
# By using the $(\nabla \times \mathbf{A}) \cdot \mathbf{B}=\mathbf{A}
# \cdot(\nabla \times \mathbf{B})+\nabla \cdot(\mathbf{A}
# \times \mathbf{B})$
# relation, we can change the first term into:
#
# $$
# \begin{align}
# & \iint_{\Omega}-\nabla \cdot(\nabla\times\mathbf{E}_s \times
# \bar{\mathbf{V}})-\nabla \times \mathbf{E}_s \cdot \nabla
# \times\bar{\mathbf{V}}+\varepsilon_{r} k_{0}^{2} \mathbf{E}_s
# \cdot \bar{\mathbf{V}}+k_{0}^{2}\left(\varepsilon_{r}-\varepsilon_b\right)
# \mathbf{E}_b \cdot \bar{\mathbf{V}} d S \\ +&\int_{\partial \Omega}
# \mathbf{n} \times \nabla \times \mathbf{E}_s \cdot \bar{\mathbf{V}}
# +\left(j n_bk_{0}+\frac{1}{2r}\right) \mathbf{n} \times \mathbf{E}_s
# \times \mathbf{n} \cdot \bar{\mathbf{V}} d l=0
# \end{align}
# $$
#
# using the divergence theorem $\iint_\Omega\nabla\cdot\mathbf{F}dS =
# \int_{\partial\Omega} \mathbf{F}\cdot\mathbf{n}dl$, we can write:
#
# $$
# \begin{align}
# & \iint_{\Omega}-\nabla \times \mathbf{E}_s \cdot \nabla \times
# \bar{\mathbf{V}}+\varepsilon_{r} k_{0}^{2} \mathbf{E}_s \cdot
# \bar{\mathbf{V}}+k_{0}^{2}\left(\varepsilon_{r}-\varepsilon_b\right)
# \mathbf{E}_b \cdot \bar{\mathbf{V}} d S \\ +&\int_{\partial \Omega}
# -(\nabla\times\mathbf{E}_s \times \bar{\mathbf{V}})\cdot\mathbf{n}
# + \mathbf{n} \times \nabla \times \mathbf{E}_s \cdot \bar{\mathbf{V}}
# +\left(j n_bk_{0}+\frac{1}{2r}\right) \mathbf{n} \times \mathbf{E}_s
# \times \mathbf{n} \cdot \bar{\mathbf{V}} d l=0
# \end{align}
# $$
#
# We can cancel $-(\nabla\times\mathbf{E}_s \times \bar{\mathbf{V}})
# \cdot\mathbf{n}$  and $\mathbf{n} \times \nabla \times \mathbf{E}_s
# \cdot \bar{\mathbf{V}}$ thanks to the triple product rule $\mathbf{A}
# \cdot(\mathbf{B} \times \mathbf{C})=\mathbf{B} \cdot(\mathbf{C} \times
# \mathbf{A})=\mathbf{C} \cdot(\mathbf{A} \times \mathbf{B})$, arriving
# at the final weak form:
#
# $$
# \begin{align}
# & \iint_{\Omega}-\nabla \times \mathbf{E}_s \cdot \nabla \times
# \bar{\mathbf{V}}+\varepsilon_{r} k_{0}^{2} \mathbf{E}_s \cdot
# \bar{\mathbf{V}}+k_{0}^{2}\left(\varepsilon_{r}-\varepsilon_b\right)
# \mathbf{E}_b \cdot \bar{\mathbf{V}} d S \\ +&\int_{\partial \Omega}
# \left(j n_bk_{0}+\frac{1}{2r}\right) \mathbf{n} \times \mathbf{E}_s \times 
# \mathbf{n} \cdot \bar{\mathbf{V}} d l=0
# \end{align}
# $$
#
# ## Implementation
# The modules that will be used are imported:

# +
import numpy as np
import ufl
from gmsh_helpers import gmsh_model_to_mesh
from mesh_wire import generate_mesh_wire
from mpi4py import MPI
from petsc4py import PETSc
from scipy.constants import epsilon_0, mu_0
from ufl import (FacetNormal, as_vector, conj, cross, curl, inner, lhs, rhs,
                 sqrt)
from utils import calculate_analytical_efficiencies

from dolfinx import fem, io

# -

# The demo can only be run with DOLFINx complex mode.

# +
if not np.issubdtype(PETSc.ScalarType, np.complexfloating):
    print("Demo should only be executed with DOLFINx complex mode")
    exit(0)
# -

# The following function is used for defining the background field.
# The inputs to the function are the angle $\theta$, the background
# refractive index $n_b$ and the vacuum wavevector $k_0$. The 
# function returns the expression $ \mathbf{E}_b = -\sin
# \theta e^{j (k_xx+k_yy)}\hat{\mathbf{u}}_x
# + \cos\theta e^{j (k_xx+k_yy)}\hat{\mathbf{u}}_y$.

# +
class background_electric_field:

    def __init__(self, theta, n_b, k0):
        self.theta = theta
        self.k0 = k0
        self.n_b = n_b

    def eval(self, x):

        kx = self.n_b * self.k0 * np.cos(self.theta)
        ky = self.n_b * self.k0 * np.sin(self.theta)
        phi = kx * x[0] + ky * x[1]

        ax = np.sin(self.theta)
        ay = np.cos(self.theta)

        return (-ax * np.exp(1j * phi), ay * np.exp(1j * phi))
# -

# +
# Definition of the radial distance from the center


def radial_distance(x):
    return np.sqrt(x[0]**2 + x[1]**2)

# Definition of the curl for a 2d vector


def curl_2d(a):

    ay_x = a[1].dx(0)
    ax_y = a[0].dx(1)

    c = as_vector((0, 0, ay_x - ax_y))

    return c


# Constant definition
um = 10**-6  # micron
nm = 10**-9  # nanometer
pi = np.pi

# Radius of the wire and of the boundary of the domain
radius_wire = 0.050 * um
radius_dom = 1 * um

# The smaller the mesh_factor, the finer is the mesh
mesh_factor = 1.2

# Finite element degree
degree = 3

# Wavelength sweep
wl0 = 0.4 * um

# Background refractive index
n_bkg = 1.33
eps_bkg = n_bkg**2

# Mesh size inside the wire
in_wire_size = mesh_factor * 7 * nm

# Mesh size at the boundary of the wire
on_wire_size = mesh_factor * 3 * nm

# Mesh size in the vacuum
bkg_size = mesh_factor * 60 * nm

# Mesh size at the boundary
boundary_size = mesh_factor * 30 * nm

# Tags for the subdomains
au_tag = 1          # gold wire
bkg_tag = 2         # background
boundary_tag = 3    # boundary

model = generate_mesh_wire(
    radius_wire, radius_dom, in_wire_size, on_wire_size, bkg_size,
    boundary_size, au_tag, bkg_tag, boundary_tag)

mesh, cell_tags, facet_tags = gmsh_model_to_mesh(
    model, cell_data=True, facet_data=True, gdim=2)

MPI.COMM_WORLD.barrier()

# Definition of finite element for the electric field
curl_el = ufl.FiniteElement("N1curl", mesh.ufl_cell(), 3)
V = fem.FunctionSpace(mesh, curl_el)

# Wavevector of the background field
k0 = 2 * np.pi / wl0

# Angle of incidence of the background field
deg = np.pi / 180
theta = 45 * deg

# Plane wave function
f = background_electric_field(theta, n_bkg, k0)
Eb = fem.Function(V)
Eb.interpolate(f.eval)

# Function r = radial distance from the (0, 0) point
lagr_el = ufl.FiniteElement("CG", mesh.ufl_cell(), 2)
lagr_space = fem.FunctionSpace(mesh, lagr_el)
r = fem.Function(lagr_space)
r.interpolate(radial_distance)

# Definition of Trial and Test functions
Es = ufl.TrialFunction(V)
Vs = ufl.TestFunction(V)

# Definition of 3d fields for cross and curl operations
Es_3d = as_vector((Es[0], Es[1], 0))
Vs_3d = as_vector((Vs[0], Vs[1], 0))

# Measures for subdomains
dx = ufl.Measure("dx", mesh, subdomain_data=cell_tags)
ds = ufl.Measure("ds", mesh, subdomain_data=facet_tags)
dAu = dx(au_tag)
dBkg = dx(bkg_tag)
dDom = dAu + dBkg
dsbc = ds(boundary_tag)

# Normal to the boundary
n = FacetNormal(mesh)
n_3d = as_vector((n[0], n[1], 0))

# Definition of relative permittivity for Au @400nm
reps_au = -1.0782
ieps_au = 5.8089
eps_au = reps_au + ieps_au * 1j

# Definition of the relative permittivity over the whole domain
D = fem.FunctionSpace(mesh, ("DG", 0))
eps = fem.Function(D)
au_cells = cell_tags.find(au_tag)
bkg_cells = cell_tags.find(bkg_tag)
eps.x.array[au_cells] = np.full_like(
    au_cells, reps_au + ieps_au * 1j, dtype=np.complex128)
eps.x.array[bkg_cells] = np.full_like(bkg_cells, eps_bkg, dtype=np.complex128)

# Weak form
F = - inner(curl(Es), curl(Vs)) * dDom \
    + eps * k0 ** 2 * inner(Es, Vs) * dDom \
    + k0 ** 2 * (eps - eps_bkg) * inner(Eb, Vs) * dDom \
    + (1j * k0 * n_bkg + 1 / (2 * r)) \
    * inner(cross(Es_3d, n_3d), cross(Vs_3d, n_3d)) * dsbc

# Splitting in left-hand side and right-hand side
a, L = lhs(F), rhs(F)

problem = fem.petsc.LinearProblem(a, L, bcs=[], petsc_options={
                                  "ksp_type": "preonly", "pc_type": "lu"})
Eh = problem.solve()

# Total electric field E = Es + Eb
E = fem.Function(V)
E.x.array[:] = Eb.x.array[:] + Eh.x.array[:]

# ||E||
norm_func = sqrt(inner(Eh, Eh))
V_normEh = fem.FunctionSpace(mesh, lagr_el)
norm_expr = fem.Expression(norm_func, V_normEh.element.interpolation_points)
normEh = fem.Function(V_normEh)
normEh.interpolate(norm_expr)

# Save the fields as xdmf files
with io.XDMFFile(MPI.COMM_WORLD, "data/Es.xdmf", "w") as xdmf:
    xdmf.write_mesh(mesh)
    xdmf.write_function(Eh)

with io.XDMFFile(MPI.COMM_WORLD, "data/E.xdmf", "w") as xdmf:
    xdmf.write_mesh(mesh)
    xdmf.write_function(E)

with io.XDMFFile(MPI.COMM_WORLD, "data/normEs.xdmf", "w") as xdmf:
    xdmf.write_mesh(mesh)
    xdmf.write_function(normEh)

# Calculation of analytical efficiencies
q_abs_analyt, q_sca_analyt, q_ext_analyt = calculate_analytical_efficiencies(
    reps_au,
    ieps_au,
    n_bkg,
    wl0,
    radius_wire)

# Vacuum impedance
Z0 = np.sqrt(mu_0 / epsilon_0)

# Magnetic field H
Hh_3d = -1j * curl_2d(Eh) / Z0 / k0 / n_bkg

Eh_3d = as_vector((Eh[0], Eh[1], 0))
E_3d = as_vector((E[0], E[1], 0))

# Intensity of the electromagnetic fields I0 = 0.5*E0**2/Z0
# E0 = np.sqrt(ax**2 + ay**2) = 1, see background_electric_field
I0 = 0.5 / Z0

# Geometrical cross section of the wire
gcs = 2 * radius_wire

# Quantities for the calculation of efficiencies
P = 0.5 * inner(cross(Eh_3d, conj(Hh_3d)), n_3d)
Q = 0.5 * ieps_au * k0 * (inner(E_3d, E_3d)) / Z0 / n_bkg

# Normalized efficiencies
q_abs_fenics_proc = (fem.assemble_scalar(fem.form(Q * dAu)) / gcs / I0).real
# Sum results from all MPI processes
q_abs_fenics = mesh.comm.allreduce(q_abs_fenics_proc, op=MPI.SUM)

q_sca_fenics_proc = (fem.assemble_scalar(fem.form(P * dsbc)) / gcs / I0).real

# Sum results from all MPI processes
q_sca_fenics = mesh.comm.allreduce(q_sca_fenics_proc, op=MPI.SUM)

q_ext_fenics = q_abs_fenics + q_sca_fenics

err_abs = np.abs(q_abs_analyt - q_abs_fenics) / q_abs_analyt * 100
err_sca = np.abs(q_sca_analyt - q_sca_fenics) / q_sca_analyt * 100
err_ext = np.abs(q_ext_analyt - q_ext_fenics) / q_ext_analyt * 100

if MPI.COMM_WORLD.rank == 0:

    print()
    print(f"The analytical absorption efficiency is {q_abs_analyt}")
    print(f"The numerical absorption efficiency is {q_abs_fenics}")
    print(f"The error is {err_abs}%")
    print()
    print(f"The analytical scattering efficiency is {q_sca_analyt}")
    print(f"The numerical scattering efficiency is {q_sca_fenics}")
    print(f"The error is {err_sca}%")
    print()
    print(f"The analytical extinction efficiency is {q_ext_analyt}")
    print(f"The numerical extinction efficiency is {q_ext_fenics}")
    print(f"The error is {err_ext}%")
# -
# ## References
