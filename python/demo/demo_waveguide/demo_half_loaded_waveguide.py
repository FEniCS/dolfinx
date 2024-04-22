# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.13.8
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# # Electromagnetic modal analysis for a waveguide

# Copyright (C) 2022 Michele Castriotta, Igor Baratta, Jørgen S. Dokken
#
# This demo is implemented in two files, one for defining and solving
# the eigenvalue problem for a half-loaded electromagnetic waveguide
# with perfect electric conducting walls, and one for verifying if the
# numerical eigenvalues are consistent with the analytical modes of the
# problem.
#
# The demo shows how to:
#
# - Setup an eigenvalue problem for Maxwell's equations
# - Use SLEPc for solving eigenvalue problems
#

# ## Equations and problem definition
#
# In this demo, we are going to show how to solve the eigenvalue problem
# associated with a half-loaded rectangular waveguide with perfect
# electric conducting walls.
#
# First of all, let's import the modules we need for solving the
# problem:

# +
import sys

from mpi4py import MPI

import numpy as np
from analytical_modes import verify_mode

import ufl
from basix.ufl import element, mixed_element
from dolfinx import default_scalar_type, fem, io, plot
from dolfinx.fem.petsc import assemble_matrix
from dolfinx.mesh import CellType, create_rectangle, exterior_facet_indices, locate_entities

try:
    import pyvista

    have_pyvista = True
except ModuleNotFoundError:
    print("pyvista and pyvistaqt are required to visualise the solution")
    have_pyvista = False

try:
    from slepc4py import SLEPc
except ModuleNotFoundError:
    print("slepc4py is required for this demo")
    sys.exit(0)
# -

# We now define the domain. It is a rectangular domain with width $w$
# and height $h = 0.45w$, with the dielectric medium filling the
# lower-half of the domain, with a height of $d=0.5h$.

# +
w = 1
h = 0.45 * w
d = 0.5 * h
nx = 300
ny = int(0.4 * nx)

msh = create_rectangle(
    MPI.COMM_WORLD, np.array([[0, 0], [w, h]]), np.array([nx, ny]), CellType.quadrilateral
)
msh.topology.create_connectivity(msh.topology.dim - 1, msh.topology.dim)
# -

# Now we can define the dielectric permittivity $\varepsilon_r$ over the
# domain as $\varepsilon_r = \varepsilon_v = 1$ in the vacuum, and as
# $\varepsilon_r = \varepsilon_d = 2.45$ in the dielectric:

# +
eps_v = 1
eps_d = 2.45


def Omega_d(x):
    return x[1] <= d


def Omega_v(x):
    return x[1] >= d


D = fem.functionspace(msh, ("DQ", 0))
eps = fem.Function(D)

cells_v = locate_entities(msh, msh.topology.dim, Omega_v)
cells_d = locate_entities(msh, msh.topology.dim, Omega_d)

eps.x.array[cells_d] = np.full_like(cells_d, eps_d, dtype=default_scalar_type)
eps.x.array[cells_v] = np.full_like(cells_v, eps_v, dtype=default_scalar_type)
# -

# In order to find the weak form of our problem, the starting point are
# Maxwell's equation and the perfect electric conductor condition on the
# waveguide wall:
#
# $$
# \begin{align}
# &\nabla \times \frac{1}{\mu_{r}} \nabla \times \mathbf{E}-k_{o}^{2}
# \epsilon_{r} \mathbf{E}=0 \quad &\text { in } \Omega\\
# &\hat{n}\times\mathbf{E} = 0 &\text { on } \Gamma
# \end{align}
# $$
#
# with $k_0$ and $\lambda_0 = 2\pi/k_0$ being the wavevector and the
# wavelength, which we consider fixed at $\lambda = h/0.2$. If we focus
# on non-magnetic material only, we can also use $\mu_r=1$.
#
# Now we can assume a known dependance on $z$:
#
# $$
# \mathbf{E}(x, y, z)=\left[\mathbf{E}_{t}(x, y)+\hat{z} E_{z}(x, y)\right]
# e^{-jk_z z}
# $$
#
# where $\mathbf{E}_t$ is the component of the electric field transverse
# to the waveguide axis, and $E_z$ is the component  of the electric
# field parallel to the waveguide axis, and $k_z$ represents our complex
# propagation constant.
#
# In order to pose the problem as an eigenvalue problem, we need to make
# the following substitution:
#
# $$
# \begin{align}
# & \mathbf{e}_t = k_z\mathbf{E}_t\\
# & e_z = -jE_z
# \end{align}
# $$
#
# The final weak form can be written as:
#
# $$
# \begin{aligned}
# F_{k_z}(\mathbf{e})=\int_{\Omega} &\left(\nabla_{t} \times
# \mathbf{e}_{t}\right) \cdot\left(\nabla_{t} \times
# \bar{\mathbf{v}}_{t}\right) -k_{o}^{2} \epsilon_{r} \mathbf{e}_{t} \cdot
# \bar{\mathbf{v}}_{t} \\
# &+k_z^{2}\left[\left(\nabla_{t} e_{z}+\mathbf{e}_{t}\right)
# \cdot\left(\nabla_{t} \bar{v}_{z}+\bar{\mathbf{v}}_{t}\right)-k_{o}^{2}
# \epsilon_{r} e_{z} \bar{v}_{z}\right] \mathrm{d} x = 0
# \end{aligned}
# $$
#
# Or, in a more compact form, as:
#
# $$
# \left[\begin{array}{cc}
# A_{t t} & 0 \\
# 0 & 0
# \end{array}\right]\left\{\begin{array}{l}
# \mathbf{e}_{t} \\
# e_{z}
# \end{array}\right\}=-k_z^{2}\left[\begin{array}{ll}
# B_{t t} & B_{t z} \\
# B_{z t} & B_{z z}
# \end{array}\right]\left\{\begin{array}{l}
# \mathbf{e}_{t} \\
# e_{z}
# \end{array}\right\}
# $$
#
# A problem of this kind is known as a generalized eigenvalue problem,
# where our eigenvalues are all the possible $ -k_z^2$. For further
# details about this problem, check Jin's *The Finite Element Method in
# Electromagnetics, third edition*.
#
# To write the weak form, we need to specify our function space. For
# $\mathbf{e}_t$, we can use RTCE elements (the equivalent of Nedelec
# elements on quadrilateral cells), while for $e_z$ field we can use
# Lagrange elements. This hybrid formulation is implemented with
# `mixed_element`:

degree = 1
RTCE = element("RTCE", msh.basix_cell(), degree)
Q = element("Lagrange", msh.basix_cell(), degree)
V = fem.functionspace(msh, mixed_element([RTCE, Q]))

# Now we can define our weak form:

# +
lmbd0 = h / 0.2
k0 = 2 * np.pi / lmbd0

et, ez = ufl.TrialFunctions(V)
vt, vz = ufl.TestFunctions(V)

a_tt = (ufl.inner(ufl.curl(et), ufl.curl(vt)) - (k0**2) * eps * ufl.inner(et, vt)) * ufl.dx
b_tt = ufl.inner(et, vt) * ufl.dx
b_tz = ufl.inner(et, ufl.grad(vz)) * ufl.dx
b_zt = ufl.inner(ufl.grad(ez), vt) * ufl.dx
b_zz = (ufl.inner(ufl.grad(ez), ufl.grad(vz)) - (k0**2) * eps * ufl.inner(ez, vz)) * ufl.dx

a = fem.form(a_tt)
b = fem.form(b_tt + b_tz + b_zt + b_zz)
# -

# Let's add the perfect electric conductor conditions on the waveguide
# wall:

# +
bc_facets = exterior_facet_indices(msh.topology)
bc_dofs = fem.locate_dofs_topological(V, msh.topology.dim - 1, bc_facets)
u_bc = fem.Function(V)
with u_bc.x.petsc_vec.localForm() as loc:
    loc.set(0)
bc = fem.dirichletbc(u_bc, bc_dofs)
# -

# ## Solve the problem in SLEPc

# Now we can solve the problem with SLEPc. First of all, we need to
# assemble our $A$ and $B$ matrices with PETSc in this way:

A = assemble_matrix(a, bcs=[bc])
A.assemble()
B = assemble_matrix(b, bcs=[bc])
B.assemble()

# Now, we need to create the eigenvalue problem in SLEPc. Our problem is
# a linear eigenvalue problem, that in SLEPc is solved with the `EPS`
# module. We can initialize this solver in the following way:

eps = SLEPc.EPS().create(msh.comm)

# We can pass to `EPS` our matrices by using the `setOperators` routine:

eps.setOperators(A, B)

# If the matrices in the problem have known properties (e.g.
# hermiticity) we can use this information in SLEPc to accelerate the
# calculation with the `setProblemType` function. For this problem,
# there is no property that can be exploited, and therefore we define it
# as a generalized non-Hermitian eigenvalue problem with the
# `SLEPc.EPS.ProblemType.GNHEP` object:

eps.setProblemType(SLEPc.EPS.ProblemType.GNHEP)

# Next, we need to specify a tolerance for the iterative solver, so that
# it knows when to stop:

tol = 1e-9
eps.setTolerances(tol=tol)

# Now we need to set the eigensolver for our problem. SLEPc offers
# different built-in algorithms, and also wrappers to external
# libraries. Some of these can only solve Hermitian problems and/or
# problems with eigenvalues in a certain portion of the spectrum.
# However, the choice of the particular method to choose to solve a
# problem is a technical discussion that is out of the scope of this
# demo, and that is more comprehensively discussed in the [SLEPc
# documentation](https://slepc.upv.es/documentation/slepc.pdf). For our
# problem, we will use the Krylov-Schur method, which we can set by
# calling the `setType` function:

eps.setType(SLEPc.EPS.Type.KRYLOVSCHUR)

# In order to accelerate the calculation of our solutions, we can also
# use a so-called *spectral transformation*, a technique which maps the
# original eigenvalues into another position of the spectrum without
# affecting the eigenvectors. In our case, we can use the
# shift-and-invert transformation with the `SLEPc.ST.Type.SINVERT`
# object:

# +
# Get ST context from eps
st = eps.getST()

# Set shift-and-invert transformation
st.setType(SLEPc.ST.Type.SINVERT)
# -

# The spectral transformation needs a target value for the eigenvalues
# we are looking for. Since the eigenvalues for our problem can be
# complex numbers, we need to specify whether we are searching for
# specific values in the real part, in the imaginary part, or in the
# magnitude. In our case, we are interested in propagating modes, and
# therefore in real $k_z$. For this reason, we can specify with the
# `setWhichEigenpairs` function that our target value will refer to the
# real part of the eigenvalue, with the `SLEPc.EPS.Which.TARGET_REAL`
# object:

eps.setWhichEigenpairs(SLEPc.EPS.Which.TARGET_REAL)

# For specifying the target value, we can use the `setTarget` function.
# Even though we cannot know a good target value a priori, we can guess
# that $k_z$ will be quite close to $k_0$ in value, for instance $k_z =
# 0.5k_0^2$. Therefore, we can set a target value of $-(0.5k_0^2)$:

eps.setTarget(-((0.5 * k0) ** 2))

# Then, we need to define the number of eigenvalues we want to
# calculate. We can do this with the `setDimensions` function, where we
# specify that we are looking for just one eigenvalue:

eps.setDimensions(nev=1)

# We can finally solve the problem with the `solve` function. To gain a
# deeper insight over the simulation, we also print an output message
# from SLEPc by calling the `view` and `errorView` function:

eps.solve()
eps.view()
eps.errorView()

# Now we can get the eigenvalues and eigenvectors calculated by SLEPc
# with the following code. We also verify if the numerical $k_z$ are
# consistent with the analytical equations of the half-loaded waveguide
# modes, with the `verify_mode()` function defined in
# `analytical_modes.py`:

# +
# Save the kz
vals = [(i, np.sqrt(-eps.getEigenvalue(i))) for i in range(eps.getConverged())]

# Sort kz by real part
vals.sort(key=lambda x: x[1].real)

eh = fem.Function(V)

kz_list = []

for i, kz in vals:
    # Save eigenvector in eh
    eps.getEigenpair(i, eh.x.petsc_vec)

    # Compute error for i-th eigenvalue
    error = eps.computeError(i, SLEPc.EPS.ErrorType.RELATIVE)

    # Verify, save and visualize solution
    if error < tol and np.isclose(kz.imag, 0, atol=tol):
        kz_list.append(kz)

        # Verify if kz is consistent with the analytical equations
        assert verify_mode(kz, w, h, d, lmbd0, eps_d, eps_v, threshold=1e-4)

        print(f"eigenvalue: {-kz**2}")
        print(f"kz: {kz}")
        print(f"kz/k0: {kz / k0}")

        eh.x.scatter_forward()

        eth, ezh = eh.split()
        eth = eh.sub(0).collapse()
        ez = eh.sub(1).collapse()

        # Transform eth, ezh into Et and Ez
        eth.x.array[:] = eth.x.array[:] / kz
        ezh.x.array[:] = ezh.x.array[:] * 1j

        gdim = msh.geometry.dim
        V_dg = fem.functionspace(msh, ("DQ", degree, (gdim,)))
        Et_dg = fem.Function(V_dg)
        Et_dg.interpolate(eth)

        # Save solutions
        with io.VTXWriter(msh.comm, f"sols/Et_{i}.bp", Et_dg) as f:
            f.write(0.0)

        with io.VTXWriter(msh.comm, f"sols/Ez_{i}.bp", ezh) as f:
            f.write(0.0)

        # Visualize solutions with Pyvista
        if have_pyvista:
            V_cells, V_types, V_x = plot.vtk_mesh(V_dg)
            V_grid = pyvista.UnstructuredGrid(V_cells, V_types, V_x)
            Et_values = np.zeros((V_x.shape[0], 3), dtype=np.float64)
            Et_values[:, : msh.topology.dim] = Et_dg.x.array.reshape(
                V_x.shape[0], msh.topology.dim
            ).real

            V_grid.point_data["u"] = Et_values

            plotter = pyvista.Plotter()
            plotter.add_mesh(V_grid.copy(), show_edges=False)
            plotter.view_xy()
            plotter.link_views()
            if not pyvista.OFF_SCREEN:
                plotter.show()
            else:
                pyvista.start_xvfb()
                plotter.screenshot("Et.png", window_size=[400, 400])

        if have_pyvista:
            V_lagr, lagr_dofs = V.sub(1).collapse()
            V_cells, V_types, V_x = plot.vtk_mesh(V_lagr)
            V_grid = pyvista.UnstructuredGrid(V_cells, V_types, V_x)
            V_grid.point_data["u"] = ezh.x.array.real[lagr_dofs]
            plotter = pyvista.Plotter()
            plotter.add_mesh(V_grid.copy(), show_edges=False)
            plotter.view_xy()
            plotter.link_views()
            if not pyvista.OFF_SCREEN:
                plotter.show()
            else:
                pyvista.start_xvfb()
                plotter.screenshot("Ez.png", window_size=[400, 400])
# -
