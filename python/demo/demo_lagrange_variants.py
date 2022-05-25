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
# This demo ({download}`demo_lagrange_variants.py`) illustrates how to:
#
# - Define finite elements directly using Basix
# - Create variants of Lagrange finite elements
#
# We begin this demo by importing everything we require.

# +
import matplotlib.pylab as plt
import numpy as np

import basix
import basix.ufl_wrapper
import ufl
from dolfinx import fem, mesh
from ufl import ds, dx, grad, inner

from mpi4py import MPI
from petsc4py.PETSc import ScalarType

if np.issubdtype(ScalarType, np.complexfloating):
    print("Demo should only be executed with DOLFINx real mode")
    exit(0)

# -

# In addition to the imports seen in other demos, we also import Basix
# and its UFL wrapper directly. Basix is the element definition and
# tabulation library that is used by FEniCSx.


# ## Equispaced points vs GLL points
# The basis function of Lagrange elements are defined by placing points
# on the reference element, with each basis function equal to 1 at one
# point and 0 at all the other points.
#
# To demonstrate the influence of interpolation point position, we
# create a degree 10 element on an interval using equally spaced points,
# and plot the basis functions. We create this element using Basix's
# [`create_element`](https://docs.fenicsproject.org/basix/main/python/demo/demo_create_and_tabulate.py.html)
# function.
#
# Basix's function `element.tabulate` returns a 4-dimensional array with
# shape (derivatives, points, basis functions, value size). In this
# example, we only tabulate the 0th derivative and the value size is
# 1, so we take the slice `[0, :, :, 0]` to get a 2-dimensional array.

# +
element = basix.create_element(basix.ElementFamily.P, basix.CellType.interval, 10,
                               basix.LagrangeVariant.equispaced)

pts = basix.create_lattice(basix.CellType.interval, 200, basix.LatticeType.equispaced, True)
values = element.tabulate(0, pts)[0, :, :, 0]
if MPI.COMM_WORLD.size == 1:  # Skip this plotting in parallel
    for i in range(values.shape[1]):
        plt.plot(pts, values[:, i])
    plt.plot(element.points, [0 for _ in element.points], "ko")
    plt.ylim([-1, 6])
    plt.savefig("demo_lagrange_variants_equispaced_10.png")
    plt.clf()
# -

# ![The basis functions of a degree 10 Lagrange space defined using
# equispaced points.](demo_lagrange_variants_equispaced_10.png)
#
# The basis functions exhibit large peaks towards the ends of the
# interval. This is known as [Runge's
# phenomenon](https://en.wikipedia.org/wiki/Runge%27s_phenomenon). The
# amplitude of the peaks increases as the degree of the element is
# increased.
#
# To rectify this issue, we can create a 'variant' of a Lagrange element
# that uses the [Gauss--Lobatto--Legendre (GLL)
# points](https://en.wikipedia.org/wiki/Gaussian_quadrature#Gauss%E2%80%93Lobatto_rules)
# to define the basis functions.

# +
element = basix.create_element(basix.ElementFamily.P, basix.CellType.interval, 10,
                               basix.LagrangeVariant.gll_warped)

values = element.tabulate(0, pts)[0, :, :, 0]

if MPI.COMM_WORLD.size == 1:  # Skip this plotting in parallel
    for i in range(values.shape[1]):
        plt.plot(pts, values[:, i])
    plt.plot(element.points, [0 for _ in element.points], "ko")
    plt.ylim([-1, 6])
    plt.savefig("demo_lagrange_variants_gll_10.png")
    plt.clf()
# -

# ![The basis functions of a degree 10 Lagrange space defined using GLL
# points.](demo_lagrange_variants_gll_10.png)
#
# The points are clustered towards the endpoints of the interval, and
# the basis functions do not exhibit Runge's phenomenon.

# ## Wrapping a Basix element
# Elements created using Basix can be used directly with UFL via Basix's
# UFL wrapper.

element = basix.create_element(basix.ElementFamily.P, basix.CellType.triangle, 3,
                               basix.LagrangeVariant.gll_warped)
ufl_element = basix.ufl_wrapper.BasixElement(element)

# The UFL element `ufl_element` can be used in the same way as an
# element created directly in UFL. For example, we could [solve a
# Poisson problem](demo_poisson) using this element.

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
# To demonstrate how the choice of Lagrange variant can affect
# computed results, we will compute the error when interpolating a
# function into a finite element space. For this example, we define a
# saw tooth wave that will be interpolated into a Lagrange space.


def saw_tooth(x):
    f = 4 * abs(x - 0.43)
    for _ in range(8):
        f = abs(f - 0.3)
    return f


# We begin by interpolating the saw tooth wave with the two Lagrange
# elements, and plot the finite element interpolation.

# +
mesh = mesh.create_unit_interval(MPI.COMM_WORLD, 10)

x = ufl.SpatialCoordinate(mesh)
u_exact = saw_tooth(x[0])

for variant in [basix.LagrangeVariant.equispaced, basix.LagrangeVariant.gll_warped]:
    element = basix.create_element(basix.ElementFamily.P, basix.CellType.interval, 10, variant)
    ufl_element = basix.ufl_wrapper.BasixElement(element)
    V = fem.FunctionSpace(mesh, ufl_element)

    uh = fem.Function(V)
    uh.interpolate(lambda x: saw_tooth(x[0]))

    if MPI.COMM_WORLD.size == 1:  # Skip this plotting in parallel
        pts = []
        cells = []
        for cell in range(10):
            for i in range(51):
                pts.append([cell / 10 + i / 50 / 10, 0, 0])
                cells.append(cell)
        values = uh.eval(pts, cells)
        plt.plot(pts, [saw_tooth(i[0]) for i in pts], "k--")
        plt.plot(pts, values, "r-")

        plt.legend(["function", "approximation"])
        plt.ylim([-0.1, 0.4])
        plt.title(variant.name)

        plt.savefig(f"demo_lagrange_variants_interpolation_{variant.name}.png")
        plt.clf()
# -

# ![](demo_lagrange_variants_interpolation_equispaced.png)
# ![](demo_lagrange_variants_interpolation_gll_warped.png)
#
# The plots illustrate that Runge's phenomenon leads to the
# interpolation being less accurate when using the equispaced variant of
# Lagrange compared to the GLL variant.
#
# To quantify the error, we compute the interpolation error in the $L_2$
# norm,
#
# $$\left\|u - u_h\right\|_2 = \left(\int_0^1 (u - u_h)^2\right)^{\frac{1}{2}},$$
#
# where $u$ is the function and $u_h$ is its interpolation in the finite
# element space. The following code uses UFL to compute the $L_2$ error for
# the equispaced and GLL variants. The $L_2$ error for the GLL variant is
# considerably smaller than the error for the equispaced variant.

# +
for variant in [basix.LagrangeVariant.equispaced, basix.LagrangeVariant.gll_warped]:
    element = basix.create_element(basix.ElementFamily.P, basix.CellType.interval, 10, variant)
    ufl_element = basix.ufl_wrapper.BasixElement(element)
    V = fem.FunctionSpace(mesh, ufl_element)

    uh = fem.Function(V)
    uh.interpolate(lambda x: saw_tooth(x[0]))
    M = fem.form((u_exact - uh)**2 * dx)
    error = mesh.comm.allreduce(fem.assemble_scalar(M), op=MPI.SUM)

    print(f"Computed L2 interpolation error ({variant.name}):", error ** 0.5)
# -

# ## Available Lagrange variants
#
# Basix supports numerous Lagrange variants, including:
#
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
#
# ### Equispaced points
# The variant `basix.LagrangeVariant.equispaced` defines an element
# using equally spaced points on the cell.
#
# ### GLL points
# For intervals, quadrilaterals and hexahedra, the variants
# `basix.LagrangeVariant.gll_warped`, `basix.LagrangeVariant.gll_isaac`
# and `basix.LagrangeVariant.gll_centroid` all define an element using
# GLL-type points.
#
# On triangles and tetrahedra, the three variants use different methods
# to distribute points on the cell so that the points on each edge are
# GLL points. The three methods used are described in [the Basix
# documentation](https://docs.fenicsproject.org/basix/main/cpp/namespacebasix_1_1lattice.html).
#
# ### Chebyshev points
# The variants `basix.LagrangeVariant.chebyshev_warped`,
# `basix.LagrangeVariant.chebyshev_isaac` and
# `basix.LagrangeVariant.chebyshev_centroid` can be used to define
# elements using [Chebyshev
# points](https://en.wikipedia.org/wiki/Chebyshev_nodes). As with GLL
# points, these three variants are the same on intervals, quadrilaterals
# and hexahedra, and vary on simplex cells.
#
# ### GL points
# The variants `basix.LagrangeVariant.gl_warped`,
# `basix.LagrangeVariant.gl_isaac` and
# `basix.LagrangeVariant.gl_centroid` can be used to define elements
# using [Gauss-Legendre (GL)
# points](https://en.wikipedia.org/wiki/Gaussian_quadrature#Gauss%E2%80%93Legendre_quadrature).
# GL points do not include the endpoints, hence this variant can only be
# used for discontinuous elements.
#
# ### Legendre polynomials
# The variant `basix.LagrangeVariant.legendre` can be used to define a
# Lagrange-like element whose basis functions are the orthonormal
# Legendre polynomials. These polynomials are not defined using points
# at the endpoints, so can also only be used for discontinuous elements.
