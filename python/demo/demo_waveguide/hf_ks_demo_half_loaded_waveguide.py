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

# # Mode analysis for a half-loaded rectangular waveguide

# INFO: I've slightly changed the code to put all the inputs of the problem
# in the same place.
# Copyright (C) 2022 Michele Castriotta, Igor Baratta, Jørgen S. Dokken
#
# This demo is implemented in one file, which shows how to:
#
# - Setup an eigenvalue problem for Maxwell's equations
# - Use SLEPc for solving eigenvalue problems in DOLFINx
#

# ## Equations, problem definition and implementation
#
# In this demo, we are going to show how to find the electromagnetic modes
# of a half-loaded rectangular waveguides.
#
# First of all, let's import the modules we need for solving the problem:

# +
import numpy as np
from slepc4py import SLEPc

import ufl
from dolfinx import fem, io
from dolfinx.mesh import (CellType, create_rectangle, exterior_facet_indices,
                          locate_entities)

from mpi4py import MPI
from petsc4py.PETSc import ScalarType
# -

# Let's now define our domain. It is a rectangular domain  with length $l$ and height $h = l/2$, while the dielectric fills the lower-half of the domain, with a height of $d=b/2$.

# +
degree = 1

l = 1
h = 0.45*l
d = 0.5*h
lmbd0 = h/0.2
k0 = 2*np.pi/lmbd0
nx = 300
ny = int(0.4*nx)
tol = 1e-4
max_it = 100

problem_type = SLEPc.EPS.Type.KRYLOVSCHUR
st_type = SLEPc.ST.Type.SINVERT
which_ep = SLEPc.EPS.Which.TARGET_REAL
target = -(0.4*k0)**2
nev = 1
ncv = 10*nev

domain = create_rectangle(MPI.COMM_WORLD, [[0, 0], [l, h]], [
    nx, ny], CellType.quadrilateral)

domain.topology.create_connectivity(domain.topology.dim - 1, domain.topology.dim)
# -

# Now we can define the dielectric permittivity $\varepsilon_r$
# over the domain as $\varepsilon_r = \varepsilon_v = 1$ in the vacuum, and as $\varepsilon_r = \varepsilon_d = 4$ in the dielectric:

# +
eps_v = 1
eps_d = 2.45

def Omega_d(x):
    return x[1] <= d

def Omega_v(x):
    return x[1] >= d

D = fem.FunctionSpace(domain, ("DG", 0))
eps = fem.Function(D)

cells_v = locate_entities(domain, domain.topology.dim, Omega_v)
cells_d = locate_entities(domain, domain.topology.dim, Omega_d)

eps.x.array[cells_d] = np.full_like(cells_d, eps_d, dtype=ScalarType)
eps.x.array[cells_v] = np.full_like(cells_v, eps_v, dtype=ScalarType)
# -

# Our equations are:
#
# $$
# \begin{align}
# &\nabla \times \frac{1}{\mu_{r}} \nabla \times \mathbf{E}-k_{o}^{2} \epsilon_{r} \mathbf{E}=0 \quad &\text { in } \Omega\\
# &\hat{n}\times\mathbf{E} = 0 &\text { on } \Gamma
# \end{align}
# $$
#
# where $k_0 = 2\pi f_0/c_0$. If we focus on non-magnetic material only, then $\mu_r=1$. Now we can assume a known dependance on $z$:
#
# $$
# \mathbf{E}(x, y, z)=\left[\mathbf{E}_{t}(x, y)+\hat{z} E_{z}(x, y)\right] e^{-jk_z z}
# $$
#
# where $\mathbf{E}_t$ is the component of the electric field transverse to the waveguide axis, and $E_z$ is the component  of the electric field parallel to the waveguide axis, and $k_z$ represents our complex propagation constant.
#
# In order to pose the problem as an eigenvalue problem, we need to make the following substitution:
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
# F_{k_z}(\mathbf{e})=\int_{\Omega} &\left(\nabla_{t} \times \mathbf{e}_{t}\right) \cdot\left(\nabla_{t} \times \bar{\mathbf{v}}_{t}\right) -k_{o}^{2} \epsilon_{r} \mathbf{e}_{t} \cdot \bar{\mathbf{v}}_{t} \\
# &+k_z^{2}\left[\left(\nabla_{t} e_{z}+\mathbf{e}_{t}\right) \cdot\left(\nabla_{t} \bar{v}_{z}+\bar{\mathbf{v}}_{t}\right)-k_{o}^{2} \epsilon_{r} e_{z} \bar{v}_{z}\right] \mathrm{d} x = 0
# \end{aligned}
# $$
#
# Or, in a more compact form:
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
# The problem is in the form of a generalized eigenvalue problem, where our eigenvalues are $\lambda = -k_z^2$. The problem is therefore the following one: once defined the frequency $f_0$ of the problem, find all the possible propagation constants $k_z$ sustained by the waveguide
#
# Let's now define the frequency $f_0$ of the electromagnetic wave and its
# correspondent $k_0$:

# We need to specify our elements. For $\mathbf{e}_t$ we can use the Nedelec elements, while for $e_z$ we can use the Lagrange elements. In DOLFINx, this hybrid formulation is implemented with `MixedElement`:

N1curl = ufl.FiniteElement("RTCE", domain.ufl_cell(), degree)
H1 = ufl.FiniteElement("Lagrange", domain.ufl_cell(), degree)
V = fem.FunctionSpace(domain, ufl.MixedElement(N1curl, H1))

# Now we can define our weak form:

# +
et, ez = ufl.TrialFunctions(V)
vt, vz = ufl.TestFunctions(V)

a_tt = (ufl.inner(ufl.curl(et), ufl.curl(vt)) - k0**2 * eps * ufl.inner(et, vt)) * ufl.dx
b_tt = ufl.inner(et, vt) * ufl.dx
b_tz = ufl.inner(et, ufl.grad(vz)) * ufl.dx
b_zt = ufl.inner(ufl.grad(ez), vt) * ufl.dx
b_zz = (ufl.inner(ufl.grad(ez), ufl.grad(vz)) - k0**2 * eps * ufl.inner(ez, vz)) * ufl.dx

a = fem.form(a_tt)
b = fem.form(b_tt + b_tz + b_zt + b_zz)
# -

# Let's now add the perfect electric conductor conditions to our weak form:

# +
bc_facets = exterior_facet_indices(domain.topology)

bc_dofs = fem.locate_dofs_topological(V, domain.topology.dim - 1, bc_facets)

u_bc = fem.Function(V)
with u_bc.vector.localForm() as loc:
    loc.set(0)
bc = fem.dirichletbc(u_bc, bc_dofs)
# -

# Let's now see how to solve the problem in SLEPc.
# First of all, we need to assemble our $A$ and $B$ matrices. This task can be done with PETSc:

A = fem.petsc.assemble_matrix(a, bcs=[bc])
A.assemble()
B = fem.petsc.assemble_matrix(b, bcs=[bc])
B.assemble()

# Now, we need to create the eigenvalue problem in SLEPc. Our problem is a linear eigenvalue problem, that in SLEPc is solved with the `EPS` module. We can call this module in the following way:

eps = SLEPc.EPS().create(domain.comm)

# Now we can pass to the `EPS` solver our matrices by using the `setOperators` routine:

eps.setOperators(A, B)

eps.setProblemType(SLEPc.EPS.ProblemType.GNHEP)

# SLEPc uses iterative algorithms to solve our problem,
# and therefore we need to specify the tolerance for the
# solution and the maximum number of iterations:

eps.setTolerances(tol = tol, 
                  max_it = max_it)

# Now we need to set the eigensolver for our problem, which is the algorithm we want to use to find the eigenvalues and the eigenvectors. SLEPc offers different methods, and also wrappers to external libraries. Some of these methods are only suitable for Hermitian or Generalized Hermitian problems and/or for eigenvalues in a certain portion of the spectrum. However, the choice of the method is a technical discussion that is out of the scope of this demo. For our problem, we will use the default Krylov-Schur method, which we can set by calling the `setType` function:

eps.setType(problem_type)

# Now we need to specify the spectral transformation we want to use
# to solve this problem. Spectral transormation are operators applied
# to our problem to accelerate the convergence of the eigensolver.
# In our case, we can use the shift-and-invert transformation, which
# can be applied it in this way:

# Get ST context from eps
st = eps.getST()
##
### Set shift-and-invert transformation
st.setType(st_type)

# Now we need to define a target eigenvalue for our problem. In particular, we want
# to find those $k_z$ with a real part near to $1.5k_0$ (since at page 658 of the [FEniCS book](https://fenicsproject.org/pub/book/book/fenics-book-2011-06-14.pdf)
# the dispersion plot at $340\text{MHz}$ shows that the eigenvalues are around this region).
# For this purpose, we need to
# specify in SLEPc that we are looking for the real part of the eigenvalue with the
# `setWhichEigenpairs` function, and that
# the target value for the real part is $-(1.5k_0)^2$ (remember that $\lambda = -k_z^2$),
# by using the `setTarget` function:

eps.setWhichEigenpairs(which_ep)
eps.setTarget(target)

# Then, we need to define the number of eigenvalues we want to calculate.
# We can do this with the `setDimensions` function, where we specify a
# number of $4$ eigenvalues:

#eps.setDimensions(nev=nev, ncv=ncv)
eps.setDimensions(nev=nev, ncv=ncv)
#eps.setMonitor(eps.EPS_MONITOR)
eps.setFromOptions()

# We can finally solve the problem and get the solutions

# +
eps.solve()
eps.view()
eps.errorView()

vals = [(i, np.sqrt(-eps.getEigenvalue(i))) for i in range(eps.getConverged())]

vals.sort(key=lambda x: x[1].real)

j = 0
eh = fem.Function(V)

for i, _ in vals:

    eignvl = eps.getEigenpair(i, eh.vector)
    error = eps.computeError(i, SLEPc.EPS.ErrorType.RELATIVE)
    
    kz = np.sqrt(-eignvl)

    if error < 1e-10:
        print()
        print(f"error: {error}")
        print(f"kz/k0: {kz/k0:.12f}")
        print()
        
        eh.name = f"E-{j:03d}-{kz/k0:.4f}"
        j += 1

        eh.x.scatter_forward()

        eth, ezh = eh.split()
        V_dg = fem.VectorFunctionSpace(domain, ("DG", degree))
        E_dg = fem.Function(V_dg)
        E_dg.interpolate(eth)
        padded_j = str(j).zfill(3)
        padded_e = str(eignvl).zfill(3)
        with io.VTXWriter(domain.comm, f"sols/E_{padded_j}_eigen{(kz/k0).real:.5f}.bp", E_dg) as f:
            f.write(0.0)
# -

# ## Analytical solutions
#
# For finding the analytical solutions to the problem, we can
# follow the formulation in *Time-harmonic electromagnetic fields* by
# Harrington. In particular, the author starts by considering the
# $\text{TE}_x$ and $\text{TM}_x$ modes, and then find transcendental
# equations for finding the $k_z$ wavevectors sustained by the structure
# at a certain frequency. If we label the dielectric region with $1$ and
# the vacuum region with $2$, the set of equations for
# the different $k_z$ is given by:
#
#
# $$
# \begin{aligned}
# \textrm{For TE}_x \textrm{ modes}:
# \begin{cases}
# &k_{x 1}{ }^{2}+\left(\frac{n \pi}{b}\right)^{2}+k_{z}{ }^{2}=k_{1}^{2}=\omega^{2} \varepsilon_{1} \\
# &k_{x 2}{ }^{2}+\left(\frac{n \pi}{b}\right)^{2}+k_{z}{ }^{2}=k_{2}{ }^{2}=\omega^{2} \varepsilon_{2} \\
# & k_{x 1} \cot k_{x 1} d=-k_{x 2} \cot \left[k_{x 2}(a-d)\right]
# \end{cases}
# \end{aligned}
# $$
#
# $$
# \begin{aligned}
# \textrm{For TM}_x \textrm{ modes}:
# \begin{cases}
# &k_{x 1}{ }^{2}+\left(\frac{n \pi}{b}\right)^{2}+k_{z}{ }^{2}=k_{1}^{2}=\omega^{2} \varepsilon_{1} \\
# &k_{x 2}{ }^{2}+\left(\frac{n \pi}{b}\right)^{2}+k_{z}{ }^{2}=k_{2}{ }^{2}=\omega^{2} \varepsilon_{2} \\
# & \frac{k_{x 1}}{\varepsilon_{1}} \tan k_{x 1} d=-\frac{k_{x 2}}{\varepsilon_{2}} \tan \left[k_{x 2}(a-d)\right]
# \end{cases}
# \end{aligned}
# $$
#
