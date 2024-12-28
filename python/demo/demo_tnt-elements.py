# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.13.6
# ---

# # Creating TNT elements using Basix's custom element interface
#
# Basix provides numerous finite elements, but there are many other
# possible elements a user may want to use. This demo
# ({download}`demo_tnt-elements.py`) shows how the Basix custom element
# interface can be used to define elements. More detailed information
# about the inputs needed to create a custom element can be found in
# [the Basix
# documentation](https://docs.fenicsproject.org/basix/main/python/demo/demo_custom_element.py.html).
#
# We begin this demo by importing the required modules.

import importlib.util

if importlib.util.find_spec("petsc4py") is not None:
    import dolfinx

    if not dolfinx.has_petsc:
        print("This demo requires DOLFINx to be compiled with PETSc enabled.")
        exit(0)
else:
    print("This demo requires petsc4py.")
    exit(0)

from mpi4py import MPI

# +
import matplotlib as mpl
import matplotlib.pylab as plt
import numpy as np

import basix
import basix.ufl
from dolfinx import default_real_type, fem, mesh
from dolfinx.fem.petsc import LinearProblem
from ufl import SpatialCoordinate, TestFunction, TrialFunction, cos, div, dx, grad, inner, sin

mpl.use("agg")
# -

# ## Defining a degree 1 TNT element
#
# We will define [tiniest tensor
# (TNT)](https://defelement.com/elements/tnt.html) elements on a
# quadrilateral ([Commuting diagrams for the TNT elements on cubes
# (Cockburn, Qiu,
# 2014)](https://doi.org/10.1090/S0025-5718-2013-02729-9)).
#
# ### The polynomial set
#
# We begin by defining a basis of the polynomial space spanned by the
# TNT element, which is defined in terms of the orthogonal Legendre
# polynomials on the cell. For a degree 2 element (here, we use the 
# superdegree rather than the conventional subdegree to align with
# the definition of other elements), the polynomial set
# contains $1$, $y$, $y^2$, $x$, $xy$, $xy^2$, $x^2$, and $x^2y$, which
# are the first 8 polynomials in the degree 2 set of polynomials on a
# quadrilateral. We create an $8 \times 9$  matrix (number of dofs by
# number of polynomials in the degree 2 set) with an $8 \times 8$
# identity in the first 8 columns. The order in which polynomials appear
# in the polynomial sets for each cell can be found in the [Basix
# documentation](https://docs.fenicsproject.org/basix/main/polyset-order.html).

wcoeffs = np.eye(8, 9)

# For elements where the coefficients matrix is not an identity, we can
# use the properties of orthonormal polynomials to compute `wcoeffs`.
# Let $\{q_0, q_1,\dots\}$ be the orthonormal polynomials of a given
# degree for a given cell, and suppose that we're trying to represent a function
# $f_i\in\operatorname{span}\{q_1, q_2,\dots\}$ (as $\{f_0, f_1,\dots\}$ is a
# basis of the polynomial space for our element). Using the properties of
# orthonormal polynomials, we see that
# $f_i = \sum_j\left(\int_R f_iq_j\,\mathrm{d}\mathbf{x}\right)q_j$,
# and so the coefficients are given by
# $a_{ij}=\int_R f_iq_j\,\mathrm{d}\mathbf{x}$.
# Hence we could compute `wcoeffs` as follows:

# +
wcoeffs2 = np.empty((8, 9))
pts, wts = basix.make_quadrature(basix.CellType.quadrilateral, 4)
evals = basix.tabulate_polynomials(
    basix.PolynomialType.legendre, basix.CellType.quadrilateral, 2, pts
)

for j, v in enumerate(evals):
    wcoeffs2[0, j] = sum(v * wts)  # 1
    wcoeffs2[1, j] = sum(v * pts[:, 1] * wts)  # y
    wcoeffs2[2, j] = sum(v * pts[:, 1] ** 2 * wts)  # y^2
    wcoeffs2[3, j] = sum(v * pts[:, 0] * pts[:, 1] * wts)  # xy
    wcoeffs2[4, j] = sum(v * pts[:, 0] * pts[:, 1] ** 2 * wts)  # xy^2
    wcoeffs2[5, j] = sum(v * pts[:, 0] ** 2 * pts[:, 1] * wts)  # x^2y
# -

# ### Interpolation operators
#
# We provide the information that defines the DOFs associated with each
# sub-entity of the cell. First, we associate a point evaluation with
# each vertex.

# +
geometry = basix.geometry(basix.CellType.quadrilateral)
topology = basix.topology(basix.CellType.quadrilateral)
x = [[], [], [], []]  # type: ignore [var-annotated]
M = [[], [], [], []]  # type: ignore [var-annotated]

for v in topology[0]:
    x[0].append(np.array(geometry[v]))
    M[0].append(np.array([[[[1.0]]]]))
# -

# For each edge, we define points and a matrix that represent the
# integral of the function along that edge. We do this by mapping
# quadrature points to the edge and putting quadrature points in the
# matrix.

# +
pts, wts = basix.make_quadrature(basix.CellType.interval, 2)
for e in topology[1]:
    v0 = geometry[e[0]]
    v1 = geometry[e[1]]
    edge_pts = np.array([v0 + p * (v1 - v0) for p in pts])
    x[1].append(edge_pts)

    mat = np.zeros((1, 1, pts.shape[0], 1))
    mat[0, 0, :, 0] = wts
    M[1].append(mat)
# -

# There are no DOFs associated with the interior of the cell for the
# lowest order TNT element, so we associate an empty list of points and
# an empty matrix with the interior.

x[2].append(np.zeros([0, 2]))
M[2].append(np.zeros([0, 1, 0, 1]))

# ### Creating the Basix element
#
# We now create the element. Using the Basix UFL interface, we can wrap
# this element so that it can be used with FFCx/DOLFINx.

tnt_degree2 = basix.ufl.custom_element(
    basix.CellType.quadrilateral,
    [],
    wcoeffs,
    x,
    M,
    0,
    basix.MapType.identity,
    basix.SobolevSpace.H1,
    False,
    1,
    2,
    dtype=default_real_type,
)

# ## Creating higher degree TNT elements
#
# The following function follows the same method as above to define
# arbitrary degree TNT elements.


def create_tnt_quadrilateral_0Form(degree):
    assert degree > 0
    # Polyset
    ndofs = 4*degree + max(degree-2,0)**2
    npoly = (degree + 1) ** 2

    wcoeffs = np.zeros((ndofs, npoly))

    dof_n = 0
    for i in range(degree):
        for j in range(degree):
            wcoeffs[dof_n, i * (degree + 1) + j] = 1
            dof_n += 1

    for i, j in [(degree, 1), (degree, 0), (0, degree)]:
        wcoeffs[dof_n, i * (degree + 1) + j] = 1
        dof_n += 1

    if degree > 1:
        wcoeffs[dof_n, 2 * degree + 1] = 1

    # Interpolation
    geometry = basix.geometry(basix.CellType.quadrilateral)
    topology = basix.topology(basix.CellType.quadrilateral)
    x = [[], [], [], []]
    M = [[], [], [], []]

    # Vertices
    for v in topology[0]:
        x[0].append(np.array(geometry[v]))
        M[0].append(np.array([[[[1.0]]]]))

    # Edges
    if degree < 2:
        for _ in topology[1]:
            x[1].append(np.zeros([0, 2]))
            M[1].append(np.zeros([0, 1, 0, 1]))
    else:
        pts, wts = basix.make_quadrature(basix.CellType.interval, 2 * degree - 2)
        poly = basix.tabulate_polynomials(
            basix.PolynomialType.legendre, basix.CellType.interval, degree - 2, pts
        )
        edge_ndofs = poly.shape[0]
        for e in topology[1]:
            x[1].append(np.array(geometry[e[0]] + np.dot(pts,[geometry[e[1]]-geometry[e[0]]])))
            mat = np.zeros((edge_ndofs, 1, len(pts), 1))
            for i in range(edge_ndofs):
                mat[i, 0, :, 0] = wts[:] * poly[i, :]
            M[1].append(mat)

    # Interior
    if degree < 3:
        x[2].append(np.zeros([0, 2]))
        M[2].append(np.zeros([0, 1, 0, 1]))
    else:
        pts, wts = basix.make_quadrature(basix.CellType.quadrilateral, 2 * degree - 2)
        u = pts[:, 0]
        v = pts[:, 1]
        pol_set = basix.polynomials.tabulate_polynomial_set(
            basix.CellType.quadrilateral, basix.PolynomialType.legendre, degree - 3, 2, pts
        )
        # this assumes the conventional [0 to 1][0 to 1] domain of the reference element, 
        # and takes the Laplacian of (1-u)*u*(1-v)*v*pol_set[0], 
        # cf https://github.com/mscroggs/symfem/blob/main/symfem/elements/tnt.py
        poly = (pol_set[5]+pol_set[3])*(u-1)*u*(v-1)*v+ \
                2*(pol_set[2]*(u-1)*u*(2*v-1)+pol_set[1]*(v-1)*v*(2*u-1)+ \
                   pol_set[0]*((u-1)*u+(v-1)*v))
        face_ndofs = poly.shape[0]
        x[2].append(pts)
        mat = np.zeros((face_ndofs, 1, len(pts), 1))
        for i in range(face_ndofs):
            mat[i, 0, :, 0] = wts[:] * poly[i, :]
        M[2].append(mat)

    return basix.ufl.custom_element(
        basix.CellType.quadrilateral,
        [],
        wcoeffs,
        x,
        M,
        0,
        basix.MapType.identity,
        basix.SobolevSpace.H1,
        False,
        max(degree - 1, 1),
        degree,
        dtype=default_real_type,
    )


# ## Comparing TNT elements, Q and S elements
#
# We now use the code above to compare TNT elements and
# [Q and S](https://defelement.com/elements/lagrange.html) elements on
# quadrilaterals. The following function takes a DOLFINx function space
# as input, and solves a Poisson problem and returns the $L_2$ error of
# the solution.


def poisson_error(V: fem.FunctionSpace):
    msh = V.mesh
    u, v = TrialFunction(V), TestFunction(V)

    x = SpatialCoordinate(msh)
    u_exact = sin(10 * x[1]) * cos(15 * x[0])
    f = -div(grad(u_exact))

    a = inner(grad(u), grad(v)) * dx
    L = inner(f, v) * dx

    # Create Dirichlet boundary condition
    u_bc = fem.Function(V)
    u_bc.interpolate(lambda x: np.sin(10 * x[1]) * np.cos(15 * x[0]))

    msh.topology.create_connectivity(msh.topology.dim - 1, msh.topology.dim)
    bndry_facets = mesh.exterior_facet_indices(msh.topology)
    bdofs = fem.locate_dofs_topological(V, msh.topology.dim - 1, bndry_facets)
    bc = fem.dirichletbc(u_bc, bdofs)

    # Solve
    problem = LinearProblem(a, L, bcs=[bc], petsc_options={"ksp_rtol": 1e-12})
    uh = problem.solve()

    M = (u_exact - uh) ** 2 * dx
    M = fem.form(M)
    error = msh.comm.allreduce(fem.assemble_scalar(M), op=MPI.SUM)
    return error**0.5


# We create a mesh, then solve the Poisson problem using our TNT
# elements of degree 1 to 9. We then do the same with Q elements of
# degree 1 to 9.

# 0-Form S elements have the following polynomial power pattern for a parallelogram or quadrilateral:
# 00000
# 0000
# 000
# 00
# 0
# (For serendipity on general quadrilateral, precision is lost, unless one uses "direct" serendipity.)

# 0-Form TNT element have the following polynomial power pattern for a parallelogram or quadrilateral:
# 00000
# 00000
# 0000
# 0000
# 00

# 0-Form Q elements have the following polynomial power pattern for a parallelogram or quadrilateral:
# 00000
# 00000
# 00000
# 00000
# 00000

# +
msh = mesh.create_unit_square(MPI.COMM_WORLD, 15, 15, mesh.CellType.quadrilateral)

tnt_ndofs = []
tnt_degrees = []
tnt_errors = []

V = fem.functionspace(msh, tnt_degree2)
#tnt_degrees.append(2)
#tnt_ndofs.append(V.dofmap.index_map.size_global)
#tnt_errors.append(poisson_error(V))
#print(f"TNT degree 2 error: {tnt_errors[-1]}")
for degree in range(1, 9):
    V = fem.functionspace(msh, create_tnt_quadrilateral_0Form(degree))
    tnt_degrees.append(degree)
    tnt_ndofs.append(V.dofmap.index_map.size_global)
    tnt_errors.append(poisson_error(V))
    print(f"TNT degree {degree} error: {tnt_errors[-1]}")

q_ndofs = []
q_degrees = []
q_errors = []
for degree in range(1, 9):
    V = fem.functionspace(msh, ("Q", degree))
    q_degrees.append(degree)
    q_ndofs.append(V.dofmap.index_map.size_global)
    q_errors.append(poisson_error(V))
    print(f"Q degree {degree} error: {q_errors[-1]}")

s_ndofs = []
s_degrees = []
s_errors = []
for degree in range(1, 9):
    V = fem.functionspace(msh, ("S", degree))
    s_degrees.append(degree)
    s_ndofs.append(V.dofmap.index_map.size_global)
    s_errors.append(poisson_error(V))
    print(f"S degree {degree} error: {s_errors[-1]}")

# -

# We now plot the data that we have obtained. First we plot the error
# against the polynomial degree for the two elements. The two elements
# appear to perform equally well.

if MPI.COMM_WORLD.rank == 0:  # Only plot on one rank
    plt.plot(q_degrees, q_errors, "bo-")
    plt.plot(tnt_degrees, tnt_errors, "gs-")
    plt.plot(s_degrees, s_errors, "rD-")
    plt.yscale("log")
    plt.xlabel("Polynomial degree")
    plt.ylabel("Error")
    plt.legend(["Q", "TNT", "S"])
    plt.savefig("demo_tnt-elements_degrees_vs_error.png")
    plt.clf()

# ![](demo_tnt-elements_degrees_vs_error.png)
#
# A key advantage of TNT elements is that for a given degree, they span
# a smaller polynomial space than Q tensorproduct elements. This can be observed in
# the following diagram, where we plot the error against the square root
# of the number of DOFs (providing a measure of cell size in 2D)
# S serendipity element perform even better here.

if MPI.COMM_WORLD.rank == 0:  # Only plot on one rank
    plt.plot(np.sqrt(q_ndofs), q_errors, "bo-")
    plt.plot(np.sqrt(tnt_ndofs), tnt_errors, "gs-")
    plt.plot(np.sqrt(s_ndofs), s_errors, "rD-")
    plt.yscale("log")
    plt.xlabel("Square root of number of DOFs")
    plt.ylabel("Error")
    plt.legend(["Q", "TNT", "S"])
    plt.savefig("demo_tnt-elements_ndofs_vs_error.png")
    plt.clf()

# ![](demo_tnt-elements_ndofs_vs_error.png)

# We can also generate the 1Form version:

def cross2d(x):
    return [x[1],-x[0]] 

def create_tnt_quadrilateral_1Form(degree):
    assert degree > 0
    # Polyset
    ndofs = 4*degree+degree**2-1+max(0,(degree-2))**2
    npoly = 2*(degree+1)**2

    wcoeffs = np.zeros((ndofs,npoly))
    for i in range(degree):
        for j in range(degree):
            wcoeffs[i*degree+j,i*(degree+1)+j]=1
            wcoeffs[degree**2+i*degree+j,(degree+1)**2+i*(degree+1)+j]=1
    wcoeffs[2*degree**2,degree**2+degree-1]=1
    wcoeffs[2*degree**2,2*(degree+1)**2-2]=-1
    
    pts, wts = basix.make_quadrature(basix.CellType.quadrilateral, 2*degree)
    poly = basix.polynomials.tabulate_polynomial_set(basix.CellType.quadrilateral, basix.PolynomialType.legendre, degree, 1, pts)
#    u = pts[:, 0]
#    v = pts[:, 1]
    for i in range((degree+1)**2):
# alternative calculation of this row for checking
#        wcoeffs[2*degree**2,i]              =sum( v*poly[0,(degree+1)**2-degree-3, :] * poly[0,i, :] * wts)
#        wcoeffs[2*degree**2,(degree+1)**2+i]=sum(-u*poly[0,(degree+1)**2-degree-3, :] * poly[0,i, :] * wts)
        wcoeffs[2*degree**2+1,i]               = sum( poly[1,2*degree+1, :] * poly[0,i, :] * wts)
        wcoeffs[2*degree**2+1,(degree+1)**2+i] = sum( poly[2,2*degree+1, :] * poly[0,i, :] * wts)
        if degree>1:
            wcoeffs[2*degree**2+2,i]               = sum( poly[1,(degree+1)**2-degree, :] * poly[0,i, :] * wts)
            wcoeffs[2*degree**2+2,(degree+1)**2+i] = sum( poly[2,(degree+1)**2-degree, :] * poly[0,i, :] * wts)

    # Interpolation
    geometry = basix.geometry(basix.CellType.quadrilateral)
    topology = basix.topology(basix.CellType.quadrilateral)
    x = [[], [], [], []]
    M = [[], [], [], []]

    # Vertices
    for v in topology[0]:
        x[0].append(np.zeros([0, 2]))
        M[0].append(np.zeros([0, 2, 0, 1]))

    # Edges
    pts, wts = basix.make_quadrature(basix.CellType.interval, 2 * degree - 2 )
    poly = basix.tabulate_polynomials(basix.PolynomialType.legendre, basix.CellType.interval, degree - 1, pts)
    for e in topology[1]:
        x[1].append(geometry[e[0]] + np.dot(pts,[geometry[e[1]]-geometry[e[0]]]))
        mat0 = np.multiply.outer(geometry[e[1]]-geometry[e[0]],wts*[poly]).transpose([2,0,3,1])
        mat1=np.zeros(mat0.shape)
        mat1[:,:,:,:]=mat0
        M[1].append(mat1)

    # Faces
    if degree < 2:
        for _ in topology[2]:
            x[2].append(np.zeros([0, 2]))
            M[2].append(np.zeros([0, 2, 0, 1]))
    else:
        pts, wts = basix.make_quadrature(basix.CellType.quadrilateral, 2 * degree - 2)
        x[2].append(pts)
        pol_set = basix.polynomials.tabulate_polynomial_set(basix.CellType.quadrilateral, basix.PolynomialType.legendre, degree - 1, 1, pts)
        mat0=(wts[None,None,None,:]*[cross2d(pol_set[1:,1:])]).transpose([2,1,3,0])
        mat1=np.zeros(mat0.shape)
        mat1[:,:,:,:]=mat0
        if degree<3:
            M[2].append(mat1)
        else:
            pol_set = basix.polynomials.tabulate_polynomial_set(basix.CellType.quadrilateral, basix.PolynomialType.legendre, degree - 3, 1, pts)
            u = pts[:, 0]
            v = pts[:, 1]
            poly = cross2d(cross2d([v*(v-1)*(pol_set[1]*(u-1)*u+pol_set[0]*(2*u-1)),u*(u-1)*(pol_set[2]*(v-1)*v+pol_set[0]*(2*v-1))]))
            mat2=(wts[None,None,None,:]*[poly]).transpose([2,1,3,0])
            M[2].append(np.concatenate((mat1,mat2)))

    return basix.ufl.custom_element(
        basix.CellType.quadrilateral,
        [2],
        wcoeffs,
        x,
        M,
        0,
        basix.MapType.covariantPiola, 
        basix.SobolevSpace.HCurl,
        False,
        degree-1,
        degree,
        basix.PolysetType.standard,
    )

# extension to hexahedrons is trivial:

def create_tnt_hexahedron(degree):
    assert degree > 0
    # Polyset
    ndofs = 12 * degree - 4 + (degree + 4) * max(degree-2, 0) ** 2
    npoly = (degree + 1) ** 3
    wcoeffs = np.zeros((ndofs, npoly))

    dof_n = 0
    for i in range(degree):
        for j in range(degree):
            for k in range(degree):
                wcoeffs[dof_n, (i * (degree + 1) + j) * (degree + 1) + k] = 1
                dof_n += 1

    for i, j, k in [(degree, 0, 0), (0, degree, 0), (0, 0, degree), (1, 1, degree), (1, 0, degree), (0, 1, degree), (degree, 1, 0)]:
        wcoeffs[dof_n, (i * (degree + 1) + j) * (degree + 1) + k] = 1
        dof_n += 1

    if degree > 1:
        for i, j, k  in [(1, degree, 1), (degree, 1, 1), (degree, 0, 1), (0, degree, 1), (1, degree, 0)]:
            wcoeffs[dof_n, (i * (degree + 1) + j) * (degree + 1) + k] = 1
            dof_n += 1

    # Interpolation
    geometry = basix.geometry(basix.CellType.hexahedron)
    topology = basix.topology(basix.CellType.hexahedron)
    x = [[], [], [], []]
    M = [[], [], [], []]

    # Vertices
    for v in topology[0]:
        x[0].append(np.array(geometry[v]))
        M[0].append(np.array([[[[1.0]]]]))

    # Edges
    if degree < 2:
        for _ in topology[1]:
            x[1].append(np.zeros([0, 3]))
            M[1].append(np.zeros([0, 1, 0, 1]))
    else:
        pts, wts = basix.make_quadrature(basix.CellType.interval, 2 * degree - 2)
        poly = basix.tabulate_polynomials(
            basix.PolynomialType.legendre, basix.CellType.interval, degree - 2, pts
        )
        edge_ndofs = poly.shape[0]
        for e in topology[1]:
            x[1].append(np.array(geometry[e[0]] + np.dot(pts,[geometry[e[1]]-geometry[e[0]]])))
            mat = np.zeros((edge_ndofs, 1, len(pts), 1))
            for i in range(edge_ndofs):
                mat[i, 0, :, 0] = wts[:] * poly[i, :]
            M[1].append(mat)


    if degree < 3:
        for _ in topology[2]:
            x[2].append(np.zeros([0, 3]))
            M[2].append(np.zeros([0, 1, 0, 1]))
        x[3].append(np.zeros([0, 3]))
        M[3].append(np.zeros([0, 1, 0, 1]))
    else:
        # Faces
        ptsr, wts = basix.make_quadrature(basix.CellType.quadrilateral, 2 * degree - 2)
        pol_set = basix.polynomials.tabulate_polynomial_set(
            basix.CellType.quadrilateral, basix.PolynomialType.legendre, degree - 3, 2, ptsr
                )
        # This takes the Laplacian of (1-u)*u*(1-v)*v*pol_set[0], 
        # cf https://github.com/mscroggs/symfem/blob/main/symfem/elements/tnt.py
        u = ptsr[:, 0]
        v = ptsr[:, 1]
        poly = (pol_set[5]+pol_set[3])*(u-1)*u*(v-1)*v+ \
                2*(pol_set[2]*(u-1)*u*(2*v-1)+pol_set[1]*(v-1)*v*(2*u-1)+ \
                    pol_set[0]*((u-1)*u+(v-1)*v))
        face_ndofs = poly.shape[0]
        for f in topology[2]:
            x[2].append(np.dot(ptsr,(geometry[f[1]]-geometry[f[0]],geometry[f[2]]-geometry[f[0]]))+geometry[f[0]])
            mat = np.zeros((face_ndofs, 1, len(ptsr), 1))
            for i in range(face_ndofs):
                mat[i, 0, :, 0] = wts[:] * poly[i, :]
            M[2].append(mat)

        # Interior

        pts, wts = basix.make_quadrature(basix.CellType.hexahedron, 2 * degree - 2)
        pol_set = basix.polynomials.tabulate_polynomial_set(
            basix.CellType.hexahedron, basix.PolynomialType.legendre, degree - 3, 2, pts
        )
        u = pts[:, 0]
        v = pts[:, 1]
        w = pts[:, 2]
        # this takes as the Laplacian of (1-u)*u*(1-v)*v*(1-w)*w*pol_set[0], 
        # cf https://github.com/mscroggs/symfem/blob/main/symfem/elements/tnt.py
        poly = (pol_set[9]+pol_set[7]+pol_set[4])*(u-1)*u*(v-1)*v*(w-1)*w+ \
                2*(pol_set[3]*(u-1)*u*(v-1)*v*(2*w-1)+pol_set[2]*(u-1)*u*(w-1)*w*(2*v-1)+pol_set[1]*(v-1)*v*(w-1)*w*(2*u-1)+ \
                   pol_set[0]*((u-1)*u*(v-1)*v+(u-1)*u*(w-1)*w+(v-1)*v*(w-1)*w))
        vol_ndofs = poly.shape[0]
        x[3].append(pts)
        mat = np.zeros((vol_ndofs, 1, len(pts), 1))
        for i in range(vol_ndofs):
            mat[i, 0, :, 0] = wts[:] * poly[i, :]
        M[3].append(mat)
        
    return basix.ufl.custom_element(
        basix.CellType.hexahedron,
        [],
        wcoeffs,
        x,
        M,
        0,
        basix.MapType.identity,
        basix.SobolevSpace.H1,
        False,
        max(degree - 1,1),
        degree,
        dtype=default_real_type,
    )

def create_tnt_prism(degree):
    assert degree > 0
    # Polyset
    ndofs = 9 * degree - 3 + (round((degree + 5) * (degree - 2) / 2) + 1) * max(degree - 2, 0)
    npoly = round((degree + 1) * (degree + 2) / 2) * (degree + 1)

    wcoeffs = np.zeros((ndofs, npoly))

    dof_n = 0
    for i in range(round((degree + 1) * degree / 2)):
        for j in range(degree):
            wcoeffs[dof_n, i * (degree + 1) + j] = 1
            dof_n += 1

    for i in range(degree+1):
        for j in range(2):
            wcoeffs[dof_n, (i + round((degree + 1) * degree / 2)) * (degree + 1) + j] = 1
            dof_n += 1

    wcoeffs[dof_n, degree] = 1
    dof_n += 1

    if degree > 1:
        for i in range(1,3):
            wcoeffs[dof_n, i * (degree + 1) + degree] = 1
            dof_n += 1


    # Interpolation
    geometry = basix.geometry(basix.CellType.prism)
    topology = basix.topology(basix.CellType.prism)
    x = [[], [], [], []]
    M = [[], [], [], []]

    # Vertices
    for v in topology[0]:
        x[0].append(np.array(geometry[v]))
        M[0].append(np.array([[[[1.0]]]]))

    # Edges
    if degree < 2:
        for _ in topology[1]:
            x[1].append(np.zeros([0, 3]))
            M[1].append(np.zeros([0, 1, 0, 1]))
    else:
        pts, wts = basix.make_quadrature(basix.CellType.interval, 2 * degree - 2)
        poly = basix.tabulate_polynomials(
            basix.PolynomialType.legendre, basix.CellType.interval, degree - 2, pts
        )
        edge_ndofs = poly.shape[0]
        for e in topology[1]:
            x[1].append(np.array(geometry[e[0]] + np.dot(pts,[geometry[e[1]]-geometry[e[0]]])))
            mat = np.zeros((edge_ndofs, 1, len(pts), 1))
            for i in range(edge_ndofs):
                mat[i, 0, :, 0] = wts[:] * poly[i, :] #* np.linalg.norm(v1 - v0)
            M[1].append(mat)

    # Faces
    if degree < 3:
        for _ in topology[2]:
            x[2].append(np.zeros([0, 3]))
            M[2].append(np.zeros([0, 1, 0, 1]))
    else:
        ptsr_t, wts_t = basix.make_quadrature(basix.CellType.triangle, 2 * degree - 2)
        pol_set_t = basix.polynomials.tabulate_polynomial_set(
            basix.CellType.triangle, basix.PolynomialType.legendre, degree - 3, 2, ptsr_t
        )
        ptsr_q, wts_q = basix.make_quadrature(basix.CellType.quadrilateral, 2 * degree - 2)
        pol_set_q = basix.polynomials.tabulate_polynomial_set(
             basix.CellType.quadrilateral, basix.PolynomialType.legendre, degree - 3, 2, ptsr_q
            )
        # this assumes the conventional [0 to 1][0 to 1] domain of the reference element, 
        # and takes the Laplacian of (1-u)*u*(1-v)*v*pol_set[0] or (u+v-1)*u*v*pol_set[0], , 
        # cf https://github.com/mscroggs/symfem/blob/main/symfem/elements/tnt.py
        u = ptsr_t[:, 0]
        v = ptsr_t[:, 1]
        poly_t = (pol_set_t[5]+pol_set_t[3])*(u+v-1)*u*v+ \
                2*(pol_set_t[2]*u*(2*v+u-1)+pol_set_t[1]*v*(2*u+v-1)+ \
                   pol_set_t[0]*(u+v))
        u = ptsr_q[:, 0]
        v = ptsr_q[:, 1]
        poly_q = (pol_set_q[5]+pol_set_q[3])*(u-1)*u*(v-1)*v+ \
                2*(pol_set_q[2]*(u-1)*u*(2*v-1)+pol_set_q[1]*(v-1)*v*(2*u-1)+ \
                   pol_set_q[0]*((u-1)*u+(v-1)*v))
        for f in topology[2]:
            if geometry[f].shape[0] == 3:
                poly = poly_t
                wts = wts_t
                ptsr = ptsr_t
            else:
                poly = poly_q
                wts = wts_q
                ptsr = ptsr_q
            face_ndofs = poly.shape[0]
            x[2].append(np.dot(ptsr,(geometry[f[1]]-geometry[f[0]],geometry[f[2]]-geometry[f[0]]))+geometry[f[0]])
            mat = np.zeros((face_ndofs, 1, len(ptsr), 1))
            for i in range(face_ndofs):
                mat[i, 0, :, 0] = wts[:] * poly[i, :] #* np.linalg.norm(np.cross(geometry[f[1]]-geometry[f[0]],geometry[f[2]]-geometry[f[0]]))
            M[2].append(mat)

    # Interior
    if degree < 4:
        x[3].append(np.zeros([0, 3]))
        M[3].append(np.zeros([0, 1, 0, 1]))
    else:
        pts, wts = basix.make_quadrature(basix.CellType.prism, 2 * degree - 2)
        #The dimension of the left over space tells us we should reduce the xy space by 1 as the xy bubble is 3rd order,
        #so we are making the appopriate selection.
        sel=[]
        for i in range(round((degree - 2) * (degree - 3)/ 2)):
            for j in range(degree - 2):
                sel.append(i * (degree  - 2) + j)
        pol_set = basix.polynomials.tabulate_polynomial_set(
            basix.CellType.prism, basix.PolynomialType.legendre, degree - 3, 2, pts)[:,sel,:]
        u = pts[:, 0]
        v = pts[:, 1]
        w = pts[:, 2]    
        # this assumes the conventional [0 to 1][0 to 1] domain of the reference element, 
        # and takes the Laplacian of (u+v-1)*u*v*(w-1)*w*pol_set[0], 
        # cf https://github.com/mscroggs/symfem/blob/main/symfem/elements/tnt.py
        poly = (pol_set[9]+pol_set[7]+pol_set[4])*(u+v-1)*u*v*(w-1)*w+ \
                2*(pol_set[3]*(u+v-1)*u*v*(2*w-1)+pol_set[2]*u*(w-1)*w*(2*v+u-1)+pol_set[1]*v*(w-1)*w*(2*u+v-1)+ \
                   pol_set[0]*((u+v-1)*u*v+u*(w-1)*w+v*(w-1)*w))
        poly=pol_set[0]
        vol_ndofs = poly.shape[0]
        x[3].append(pts)
        mat = np.zeros((vol_ndofs, 1, len(pts), 1))
        for i in range(vol_ndofs):
            mat[i, 0, :, 0] = wts[:] * poly[i, :]
        M[3].append(mat)

#    print("x: ", x)
    
    return basix.ufl.custom_element(
        basix.CellType.prism,
        [],
        wcoeffs,
        x,
        M,
        0,
        basix.MapType.identity,
        basix.SobolevSpace.H1,
        False,
        max(degree - 1,1),
        degree,
        dtype=default_real_type,
    )

# Here is the matching tetrahedron, triangle and interval elements. Their boundary dofs are moments, not point values, except at the vertices.
# The performance of the triangle element 0-form matches the P type element, as the same space is spanned. The interior spaces are differently parametrized.
# The performance of the triangle element 1-Form matches the Legendre variant of Nedelec1, as the same space is spanned. The interior spaces are differently parametrized.

def create_tnt_tetrahedron(degree):
    assert degree > 0
    # Polyset
    ndofs = round((degree+3)*(degree+1)*(degree+2)/6)
    npoly = round((degree+3)*(degree+1)*(degree+2)/6)
    
    wcoeffs = np.eye(ndofs)

    # Interpolation
    geometry = basix.geometry(basix.CellType.tetrahedron)
    topology = basix.topology(basix.CellType.tetrahedron)
    x = [[], [], [], []]
    M = [[], [], [], []]

    # Vertices
    for v in topology[0]:
        x[0].append(np.array(geometry[v]))
        M[0].append(np.array([[[[1.0]]]]))

    # Edges
    if degree < 2:
        for _ in topology[1]:
            x[1].append(np.zeros([0, 3]))
            M[1].append(np.zeros([0, 1, 0, 1]))
    else:
        pts, wts = basix.make_quadrature(basix.CellType.interval, 2 * degree - 2)
        poly = basix.tabulate_polynomials(
            basix.PolynomialType.legendre, basix.CellType.interval, degree - 2, pts
        )
        edge_ndofs = poly.shape[0]
        for e in topology[1]:
            x[1].append(np.array(geometry[e[0]] + np.dot(pts,[geometry[e[1]]-geometry[e[0]]])))
            mat = np.zeros((edge_ndofs, 1, len(pts), 1))
            for i in range(edge_ndofs):
                mat[i, 0, :, 0] = wts[:] * poly[i, :]
            M[1].append(mat)

    # Faces
    if degree < 3:
        for _ in topology[2]:
            x[2].append(np.zeros([0, 3]))
            M[2].append(np.zeros([0, 1, 0, 1]))
    else:
        ptsr, wts = basix.make_quadrature(basix.CellType.triangle, 2 * degree - 2)
        pol_set = basix.polynomials.tabulate_polynomial_set(
            basix.CellType.triangle, basix.PolynomialType.legendre, degree - 3, 2, ptsr
        )
        # this assumes the conventional [0 to 1][0 to 1] domain of the reference element, 
        # and takes the Laplacian of (u+v-1)*u*v*pol_set[0], , 
        # cf https://github.com/mscroggs/symfem/blob/main/symfem/elements/tnt.py
        u = ptsr[:, 0]
        v = ptsr[:, 1]
        poly = (pol_set[5]+pol_set[3])*(u+v-1)*u*v+ \
                2*(pol_set[2]*u*(2*v+u-1)+pol_set[1]*v*(2*u+v-1)+ \
                   pol_set[0]*(u+v))

        for f in topology[2]:
            face_ndofs = poly.shape[0]
            x[2].append(np.dot(ptsr,(geometry[f[1]]-geometry[f[0]],geometry[f[2]]-geometry[f[0]]))+geometry[f[0]])
            mat = np.zeros((face_ndofs, 1, len(ptsr), 1))
            for i in range(face_ndofs):
                mat[i, 0, :, 0] = wts[:] * poly[i, :]
            M[2].append(mat)

    # Interior
    if degree < 4:
        x[3].append(np.zeros([0, 3]))
        M[3].append(np.zeros([0, 1, 0, 1]))
    else:
        pts, wts = basix.make_quadrature(basix.CellType.tetrahedron, 2 * degree - 2)
        pol_set = basix.polynomials.tabulate_polynomial_set(
            basix.CellType.tetrahedron, basix.PolynomialType.legendre, degree - 4, 2, pts)
        u = pts[:, 0]
        v = pts[:, 1]
        w = pts[:, 2]    
        # this assumes the conventional [0 to 1][0 to 1] domain of the reference element, 
        # and takes the Laplacian of u*v*w*(u+v+w-1)*pol_set[0], 
        # cf https://github.com/mscroggs/symfem/blob/main/symfem/elements/tnt.py
        poly = (pol_set[9]+pol_set[7]+pol_set[4])*u*v*w*(u+v+w-1)+ \
                2*(pol_set[3]*(2*w+u+v-1)*u*v+pol_set[2]*u*w*(2*v+u+w-1)+pol_set[1]*v*w*(2*u+v+w-1)+ \
                   pol_set[0]*(u*v+u*w+v*w))
        poly=pol_set[0]
        vol_ndofs = poly.shape[0]
        x[3].append(pts)
        mat = np.zeros((vol_ndofs, 1, len(pts), 1))
        for i in range(vol_ndofs):
            mat[i, 0, :, 0] = wts[:] * poly[i, :]
        M[3].append(mat)

    return basix.ufl.custom_element(
        basix.CellType.tetrahedron,
        [],
        wcoeffs,
        x,
        M,
        0,
        basix.MapType.identity,
        basix.SobolevSpace.H1,
        False,
        degree,
        degree,
        dtype=default_real_type,
    )
    
def create_tnt_triangle_0Form(degree):
    assert degree > 0
    # Polyset
    ndofs = round((degree+1)*(degree+2)/2)
    npoly = round((degree+1)*(degree+2)/2)
    
    wcoeffs = np.eye(ndofs)

    # Interpolation
    geometry = basix.geometry(basix.CellType.triangle)
    topology = basix.topology(basix.CellType.triangle)
    x = [[], [], [], []]
    M = [[], [], [], []]

    # Vertices
    for v in topology[0]:
        x[0].append(np.array(geometry[v]))
        M[0].append(np.array([[[[1.0]]]]))

    # Edges
    if degree < 2:
        for _ in topology[1]:
            x[1].append(np.zeros([0, 2]))
            M[1].append(np.zeros([0, 1, 0, 1]))
    else:
        pts, wts = basix.make_quadrature(basix.CellType.interval, 2 * degree - 2)
        poly = basix.tabulate_polynomials(basix.PolynomialType.legendre, basix.CellType.interval, degree - 2, pts)
        for e in topology[1]:
            x[1].append(geometry[e[0]] + np.dot(pts,[geometry[e[1]]-geometry[e[0]]]))
            M[1].append((wts[None,None,:]*[[poly]]).transpose([2,1,3,0]))
            print(geometry[e[0]] + np.dot(pts,[geometry[e[1]]-geometry[e[0]]]))

    # Faces
    if degree < 3:
        for _ in topology[2]:
            x[2].append(np.zeros([0, 2]))
            M[2].append(np.zeros([0, 1, 0, 1]))
    else:
        pts, wts = basix.make_quadrature(basix.CellType.triangle, 2 * degree - 2)
        pol_set = basix.polynomials.tabulate_polynomial_set(basix.CellType.triangle, basix.PolynomialType.legendre, degree - 3, 2, pts)
        # this assumes the conventional [0 to 1][0 to 1] domain of the reference element, 
        # and takes the Laplacian of (u+v-1)*u*v*pol_set[0], , 
        # cf https://github.com/mscroggs/symfem/blob/main/symfem/elements/tnt.py
        u = pts[:, 0]
        v = pts[:, 1]
        poly = (pol_set[5]+pol_set[3])*(u+v-1)*u*v+ \
                2*(pol_set[2]*u*(2*v+u-1)+pol_set[1]*v*(2*u+v-1)+ \
                   pol_set[0]*(u+v))

        x[2].append(pts)
        M[2].append((wts[None,None,:]*[[poly]]).transpose([2,1,3,0]))

    
    return basix.ufl.custom_element(
        basix.CellType.triangle,
        [],
        wcoeffs,
        x,
        M,
        0,
        basix.MapType.identity,
        basix.SobolevSpace.H1,
        False,
        degree,
        degree,
        dtype=default_real_type,
    )
    
def cross2d(x):
    return [x[1],-x[0]] 

def create_tnt_triangle_1Form (degree):
    assert degree > 0
    # Polyset
    ndofs = (degree+2)*degree
    npoly = (degree+1)*(degree+2)
    
    wcoeffs = np.zeros((ndofs,npoly))
    wcoeffs[:round((degree+1)*degree/2),:round((degree+1)*degree/2)] = np.eye(round((degree+1)*degree/2))
    wcoeffs[round((degree+1)*degree/2):(degree+1)*degree,round((degree+1)*(degree+2)/2):round((degree+1)*(degree+2)/2)+round((degree+1)*degree/2)] = np.eye(round((degree+1)*degree/2))

    pts, wts = basix.make_quadrature(basix.CellType.triangle, 2*degree)
    poly  = basix.tabulate_polynomials(basix.PolynomialType.legendre, basix.CellType.triangle, degree, pts)
    poly0 = basix.tabulate_polynomials(basix.PolynomialType.legendre, basix.CellType.triangle, degree-1, pts)
    u = pts[:, 0]
    v = pts[:, 1]
    for j in range(degree):
        for i in range(round((degree+1)*(degree+2)/2)):
            wcoeffs[(degree+1)*degree + j, i]                                  = sum( v * poly0[round((degree-1)*(degree)/2)+j, :] * poly[i, :] * wts)
            wcoeffs[(degree+1)*degree + j, round((degree+1)*(degree+2)/2) + i] = sum(-u * poly0[round((degree-1)*(degree)/2)+j, :] * poly[i, :] * wts)

    # Interpolation
    geometry = basix.geometry(basix.CellType.triangle)
    topology = basix.topology(basix.CellType.triangle)
    x = [[], [], [], []]
    M = [[], [], [], []]

    # Vertices
    for v in topology[0]:
        x[0].append(np.zeros([0, 2]))
        M[0].append(np.zeros([0, 2, 0, 1]))

    # Edges
    pts, wts = basix.make_quadrature(basix.CellType.interval, 2 * degree - 2 )
    poly = basix.tabulate_polynomials(basix.PolynomialType.legendre, basix.CellType.interval, degree - 1, pts)
    for e in topology[1]:
        x[1].append(geometry[e[0]] + np.dot(pts,[geometry[e[1]]-geometry[e[0]]]))
        mat0 = np.multiply.outer(geometry[e[1]]-geometry[e[0]],wts*[poly]).transpose([2,0,3,1])
        mat1=np.zeros(mat0.shape)
        mat1[:,:,:,:]=mat0
        M[1].append(mat1)
        
    # Faces
    if degree < 2:
        for _ in topology[2]:
            x[2].append(np.zeros([0, 2]))
            M[2].append(np.zeros([0, 2, 0, 1]))
    else:
        pts, wts = basix.make_quadrature(basix.CellType.triangle, 2 * degree - 2)
        x[2].append(pts)
        pol_set = basix.polynomials.tabulate_polynomial_set(basix.CellType.triangle, basix.PolynomialType.legendre, degree - 1, 1, pts)
        mat0=(wts[None,None,None,:]*[cross2d(pol_set[1:,1:])]).transpose([2,1,3,0])
        mat1=np.zeros(mat0.shape)
        mat1[:,:,:,:]=mat0
        if degree<3:
            M[2].append(mat1)
        else:
            pol_set = basix.polynomials.tabulate_polynomial_set(basix.CellType.triangle, basix.PolynomialType.legendre, degree - 3, 1, pts)
            u = pts[:, 0]
            v = pts[:, 1]
            poly = cross2d(cross2d([v*(pol_set[1]*(u+v-1)*u+pol_set[0]*(2*u+v-1)),u*(pol_set[2]*(u+v-1)*v+pol_set[0]*(2*v+u-1))]))
            mat2=(wts[None,None,None,:]*[poly]).transpose([2,1,3,0])
            M[2].append(np.concatenate((mat1,mat2)))


    return basix.ufl.custom_element(
        basix.CellType.triangle,
        [2],
        wcoeffs,
        x,
        M,
        0,
        basix.MapType.covariantPiola, 
        basix.SobolevSpace.HCurl,
        False,
        degree-1,
        degree,
        basix.PolysetType.standard,
    )
    
# The interval elements match the Legendre variant of the serendipity elements for the 0-form
# The interval elements match the Legendre variant of the discontinuous P elements for the 1-Form of 1 lower degree

def create_tnt_interval_0Form(degree):
    assert degree > 0
    # Polyset
    ndofs = degree + 1
    npoly = degree + 1
    
    wcoeffs = np.eye(ndofs)

    # Interpolation
    geometry = basix.geometry(basix.CellType.interval)
    topology = basix.topology(basix.CellType.interval)
    x = [[], [], [], []]
    M = [[], [], [], []]

    # Vertices
    for v in topology[0]:
        x[0].append(np.array(geometry[v]))
        M[0].append(np.array([[[[1.0]]]]))

    # Edges
    if degree < 2:
        for _ in topology[1]:
            x[1].append(np.zeros([0, 1]))
            M[1].append(np.zeros([0, 1, 0, 1]))
    else:
        pts, wts = basix.make_quadrature(basix.CellType.interval, 2 * degree - 2)
        poly = basix.tabulate_polynomials(
            basix.PolynomialType.legendre, basix.CellType.interval, degree - 2, pts
        )
        for e in topology[1]:
            x[1].append(np.array(geometry[e[0]] + np.dot(pts,[geometry[e[1]]-geometry[e[0]]])))    
            M[1].append((wts*[[poly]]).transpose([2,1,3,0]))
    
    return basix.ufl.custom_element(
        basix.CellType.interval,
        [],
        wcoeffs,
        x,
        M,
        0,
        basix.MapType.identity,
        basix.SobolevSpace.H1,
        False,
        degree,
        degree,
        dtype=default_real_type,
    )

def create_tnt_interval_1Form(degree):
    assert degree > 0
    # Polyset
    ndofs = degree
    npoly = degree
    
    wcoeffs = np.eye(ndofs)

    # Interpolation
    geometry = basix.geometry(basix.CellType.interval)
    topology = basix.topology(basix.CellType.interval)
    x = [[], [], [], []]
    M = [[], [], [], []]

    # Vertices
    for v in topology[0]:
        x[0].append(np.zeros([0, 1]))
        M[0].append(np.zeros([0, 1, 0, 1]))

    # Edges
    pts, wts = basix.make_quadrature(basix.CellType.interval, 2 * degree - 1)
    poly = basix.tabulate_polynomials(
        basix.PolynomialType.legendre, basix.CellType.interval, degree - 1, pts
    )
    for e in topology[1]:
        x[1].append(np.array(geometry[e[0]] + np.dot(pts,[geometry[e[1]]-geometry[e[0]]])))   
        M[1].append((wts*[[poly]]).transpose([2,1,3,0]))

    return basix.ufl.custom_element(
        basix.CellType.interval,
        [],
        wcoeffs,
        x,
        M,
        0,
        basix.MapType.identity,
        basix.SobolevSpace.H1,
        False,
        degree-1,
        degree-1,
        dtype=default_real_type,
    )
