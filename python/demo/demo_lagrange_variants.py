# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.13.6
# ---

# # Variants of Lagrange elements
#
# This demo is implemented in a single Python file,
# {download}`demo_lagrange_variants.py`. It illustrates how to:
#
# - Define finite elements directly using Basix
# - Create variants of Lagrange finite elements
#
# We begin this demo by importing everything we require.

# +
import numpy as np
import ufl

from dolfinx import fem, mesh
from ufl import ds, dx, grad, inner
from dolfinx.fem import Function, FunctionSpace, assemble_scalar, form
from dolfinx.mesh import create_unit_interval

from mpi4py import MPI
from petsc4py.PETSc import ScalarType

try:
    import matplotlib.pylab as plt
    plotting = True
except ModuleNotFoundError:
    print("")
    plotting = False
# -

# In addition to the imports seen in many earlier demos, we also import Basix and its UFL wrapper directly.
# Basix is the element definition and tabulation library that is used by FEniCSx.

import basix
import basix.ufl_wrapper

# ## Equispaced points vs GLL points
# The basis function of Lagrange elements are defined by placing a series of points on the reference element
# then setting each basis function to be equal to 1 at one point and 0 at all the others.
#
# To demonstrate why we might need to use a variant of a Lagrange element, we can create a degree 10 element
# on an interval definied using equally spaced points and plot its basis functions. We create this element
# using Basix's
# [`create_element`](https://docs.fenicsproject.org/basix/main/python/demo/demo_create_and_tabulate.py.html)
# function.

# +
element = basix.create_element(
    basix.ElementFamily.P, basix.CellType.interval, 10, basix.LagrangeVariant.equispaced)

pts = basix.create_lattice(basix.CellType.interval, 200, basix.LatticeType.equispaced, True)
values = element.tabulate(0, pts)[0, :, :, 0]
if plotting:
    for i in range(values.shape[1]):
        plt.plot(pts, values[:, i])
    plt.plot(element.points, [0 for i in element.points], "ko")
    plt.ylim([-1, 6])
    plt.savefig("img/demo_lagrange_variants_equispaced_10.png")
    plt.clf()
# -

# ![The basis functions of a degree 10 Lagrange space defined using equispaced
# points](img/demo_lagrange_variants_equispaced_10.png)
#
# The basis functions exhibit large peaks towards either end of the interval due to
# [Runge's phenomenon](https://en.wikipedia.org/wiki/Runge%27s_phenomenon). The size of the peaks near the
# ends of the interval will in general increase in size as we increase the degree of the element.
#
# To rectify this issue, we can create a variant of a Lagrange element that uses
# [Gauss--Lobatto--Legendre (GLL) points](https://en.wikipedia.org/wiki/Gaussian_quadrature#Gauss%E2%80%93Lobatto_rules)
# to define the basis functions.

# +
element = basix.create_element(
    basix.ElementFamily.P, basix.CellType.interval, 10, basix.LagrangeVariant.gll_warped)

pts = basix.create_lattice(basix.CellType.interval, 200, basix.LatticeType.equispaced, True)
values = element.tabulate(0, pts)[0, :, :, 0]
if plotting:
    for i in range(values.shape[1]):
        plt.plot(pts, values[:, i])
    plt.plot(element.points, [0 for i in element.points], "ko")
    plt.ylim([-1, 6])
    plt.savefig("img/demo_lagrange_variants_gll_10.png")
    plt.clf()
# -

# ![The basis functions of a degree 10 Lagrange space defined using GLL points](img/demo_lagrange_variants_gll_10.png)
#
# In this variant, the points are positioned more densely towards the endpoints of the interval. The
# basis functions of this variant do not exhibit Runge's phenomenon.

# ## Wrapping a Basix element
# Elements created using Basix can be used directly with UFL via Basix's UFL wrapper.

element = basix.create_element(
    basix.ElementFamily.P, basix.CellType.triangle, 3, basix.LagrangeVariant.gll_warped)
ufl_element = basix.ufl_wrapper.BasixElement(element)

# The UFL element created by this wrapped can then be used in the same was as an element created
# directly in UFL. For example, we could [solve a Poisson problem](demo_poisson) using this element.

# +
msh = mesh.create_rectangle(comm=MPI.COMM_WORLD,
                            points=((0.0, 0.0), (2.0, 1.0)), n=(32, 16),
                            cell_type=mesh.CellType.triangle,)
V = fem.FunctionSpace(msh, ufl_element)

facets = mesh.locate_entities_boundary(msh, dim=1,
                                       marker=lambda x: np.logical_or(np.isclose(x[0], 0.0),
                                                                      np.isclose(x[0], 2.0)))

dofs = fem.locate_dofs_topological(V=V, entity_dim=1, entities=facets)

bc = fem.dirichletbc(value=ScalarType(0), dofs=dofs, V=V)

u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)
x = ufl.SpatialCoordinate(msh)
f = 10 * ufl.exp(-((x[0] - 0.5) ** 2 + (x[1] - 0.5) ** 2) / 0.02)
g = ufl.sin(5 * x[0])
a = inner(grad(u), grad(v)) * dx
L = inner(f, v) * dx + inner(g, v) * ds

problem = fem.petsc.LinearProblem(a, L, bcs=[bc], petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
uh = problem.solve()
# -

# ## Computing the error of an interpolation
# To demonstrate how a choice of Lagrange variant can affect results in DOLFINx, we can look at the
# error when interpolating a function into a finite element. For this example, we define a saw tooth
# wave that we will interpolate into a Lagrange element on an interval.


def saw_tooth(x):
    f = 4 * abs(x - 0.43)
    for i in range(8):
        f = abs(f - 0.3)
    return f


# To illustrate what we are doing, we begin by interpolating the saw tooth wave into the two
# variants of Lagrange on an interval that we plotted above, and plot the approximation in
# the finite element space

# +
mesh = create_unit_interval(MPI.COMM_WORLD, 10)


def fun(x):
    return saw_tooth(x[0])


x = ufl.SpatialCoordinate(mesh)
u_exact = saw_tooth(x[0])

for variant in [basix.LagrangeVariant.equispaced, basix.LagrangeVariant.gll_warped]:
    element = basix.create_element(
        basix.ElementFamily.P, basix.CellType.interval, 10, variant)
    ufl_element = basix.ufl_wrapper.BasixElement(element)
    V = FunctionSpace(mesh, ufl_element)

    uh = Function(V)
    uh.interpolate(lambda x: fun(x))

    pts = []
    cells = []
    for cell in range(10):
        for i in range(51):
            pts.append([cell / 10 + i / 50 / 10, 0, 0])
            cells.append(cell)
    pts = np.array(pts)
    values = uh.eval(pts, cells)
    if plotting:
        plt.plot(pts[:, 0], [fun(i) for i in pts], "k--")
        plt.plot(pts[:, 0], values, "r-")

        plt.legend(["function", "approximation"])
        plt.ylim([-0.1, 0.4])
        plt.title(variant.name)

        plt.savefig(f"img/demo_lagrange_variants_interpolation_{variant.name}.png")
        plt.clf()
# -

# ![](img/demo_lagrange_variants_interpolation_equispaced.png)
# ![](img/demo_lagrange_variants_interpolation_gll_warped.png)
#
# These plots show that the spurious peaks due to Runge's phenomenon that we say above make
# another appearance and lead to the interpolation being less accurate when using the
# equispaced variant of Lagrange.
#
# To better quantify the error that we observe, we can compute the L2 error of the
# interpolation. This is given by
#
# $$\left\|u-u_h\right\|_2 = \left(\int_0^1(u-u_h)^2\right)^{\frac{1}{2}},$$
#
# where $u$ is the function and $u_h$ is its approximation in the finite element space.
# The following code snippet uses UFL to compute this for the equispaced and GLL variants.
# The L2 error for the GLL variant is approximately 0.0015. For the equispaced variant,
# the L2 error is around 10 times larger: approximately 0.016.

# +
for variant in [basix.LagrangeVariant.equispaced, basix.LagrangeVariant.gll_warped]:
    element = basix.create_element(
        basix.ElementFamily.P, basix.CellType.interval, 10, variant)
    ufl_element = basix.ufl_wrapper.BasixElement(element)
    V = FunctionSpace(mesh, ufl_element)

    uh = Function(V)
    uh.interpolate(lambda x: fun(x))

    M = (u_exact - uh)**2 * dx
    M = form(M)
    error = mesh.comm.allreduce(assemble_scalar(M), op=MPI.SUM)

    print(variant.name, error ** 0.5)
# -

# ## Available Lagrange variants
#
# Basix provides the following Lagrange variants:
#
# - `basix.LagrangeVariant.unset`
# - `basix.LagrangeVariant.equispaced`
# - `basix.LagrangeVariant.gll_warped`
# - `basix.LagrangeVariant.gll_isaac`
# - `basix.LagrangeVariant.gll_centroid`
# - `basix.LagrangeVariant.chebyshev_warped`
# - `basix.LagrangeVariant.chebyshev_isaac`
# - `basix.LagrangeVariant.chebyshev_centroid`
# - `basix.LagrangeVariant.gl_warped`
# - `basix.LagrangeVariant.gl_isaac`
# - `basix.LagrangeVariant.gl_centroid`
# - `basix.LagrangeVariant.legendre`
# - `basix.LagrangeVariant.vtk`
#
# ### `basix.LagrangeVariant.unset`
# This variant is used internally by Basix for low degree Lagrange that do not require
# a variant.
#
# ### Equispaced points
# The variant `basix.LagrangeVariant.equispaced` will define and element using equally spaced
# points on the cell.
#
# ### GLL points
# For intervals, quadrilaterals and hexahedra, the variants `basix.LagrangeVariant.gll_warped`,
# `basix.LagrangeVariant.gll_isaac` and `basix.LagrangeVariant.gll_centroid` will all define
# an element using GLL points.
#
# On triangles and tetrahedra, these three variants use different methods to distribute
# points on the cell so that the points on each edge are GLL points. The three methods used
# are described in
# [the Basix documentation](https://docs.fenicsproject.org/basix/main/cpp/namespacebasix_1_1lattice.html).
#
# ### Chebyshev points
# The variants `basix.LagrangeVariant.chebyshev_warped`,
# `basix.LagrangeVariant.chebyshev_isaac` and `basix.LagrangeVariant.chebyshev_centroid`
# can be used to definite elements using
# [Chebyshev points](https://en.wikipedia.org/wiki/Chebyshev_nodes). As with GLL points,
# these three variants are the same on intervals, quadrilaterals and hexahedra, and use
# different simplex methods on other cells.
#
# ### GL points
# The variants `basix.LagrangeVariant.gl_warped`, `basix.LagrangeVariant.gl_isaac` and
# `basix.LagrangeVariant.gl_centroid` can be used to define elements using
# [Gauss-Legendre (GL) points](https://en.wikipedia.org/wiki/Gaussian_quadrature#Gauss%E2%80%93Legendre_quadrature).
# As GL points do not include the endpoints, these variants can only be used for
# discontinuous Lagrange elements.
#
# ### Legendre polynomials
# The variant `basix.LagrangeVariant.legendre` can be used to define a Lagrange element
# whose basis functions are the orthonormal Legendre polynomials. These polynomials
# are not defined using points at the endpoints, so can also only be used for
# discontinuous Lagrange elements.
#
# ### VTK variant
# The variant `basix.LagrangeVariant.vtk` can be used to define a Lagrange element with
# points ordered to match the ordering used by VTK. This variant should only be used when
# inputting and outputting to/from VTK. Due to how Basix handles the numbering of points
# by sub-entity, this variant can only be used for discontinuous Lagrange elements.
