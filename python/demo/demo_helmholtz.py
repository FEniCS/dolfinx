# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.13.6
# ---

# # Helmholtz equation: scattering of sound by rigid obstacles
#
# Copyright (C) 2018-2026 Samuel Groth, Jørgen S. Dokken and Garth N.
# Wells
#
# ```{admonition} Download sources
# :class: download
# * {download}`Python script <./demo_helmholtz.py>`
# * {download}`Jupyter notebook <./demo_helmholtz.ipynb>`
# ```
#
# This demo illustrates how to:
# - Create a complex-valued (sesquilinear) finite element formulation
# - Generate an exterior scattering domain with [Gmsh](https://gmsh.info),
#   using curved (second-order) cells and a mesh size field that refines
#   towards the scatterers
# - Apply a sound-hard (Neumann) condition on a scatterer and an
#   approximate radiation condition on the truncation boundary
# - Verify a computed field against an analytical (Mie series) solution
# - Assemble and solve a complex-valued problem in a DOLFINx build with
#   real PETSc scalars
# - Build a two-level overlapping Schwarz preconditioner that runs in
#   parallel, with impedance transmission between subdomains and a
#   Dirichlet-to-Neumann coarse space
#
# The demo requires Gmsh.
#
# ## Problem definition
#
# A time-harmonic acoustic pressure $p(\boldsymbol{x}, t) = \Re \left(u
# (\boldsymbol{x}) e^{-i \omega t} \right)$ of angular frequency $\omega$
# in a medium with sound speed $c$ has a complex amplitude $u$ that
# satisfies the Helmholtz equation
#
# $$
# \nabla^2 u + k^2 u = 0, \qquad k = \omega / c = 2 \pi / \lambda,
# $$
#
# where $k$ is the wavenumber and $\lambda$ the wavelength. An incident
# plane wave $u_{i} = e^{i k \boldsymbol{d} \cdot \boldsymbol{x}}$
# travelling in the unit direction $\boldsymbol{d}$ strikes a collection
# of rigid (sound-hard) obstacles with boundary $\Gamma$, and is
# scattered. We solve for the scattered field $u_{s} = u - u_{i}$, which
# is the part of the solution that radiates outwards. On a sound-hard
# obstacle the normal velocity vanishes, $\partial u / \partial n = 0$,
# hence
#
# $$
# \frac{\partial u_{s}}{\partial n} = - \frac{\partial u_{i}}{\partial n}
# = - i k (\boldsymbol{d} \cdot \boldsymbol{n}) u_{i} \quad
# \text{on } \Gamma.
# $$
#
# The scattered field is defined on an unbounded domain and satisfies the
# Sommerfeld radiation condition at infinity. To get a finite
# computational domain $\Omega$ we truncate at a circle $\Gamma_{R}$ of
# radius $R$ centred at the origin, and impose there the first-order
# Bayliss-Gunzburger-Turkel condition {cite}`bayliss1982exterior`
#
# $$
# \frac{\partial u_{s}}{\partial n} = \left(i k -
# \frac{1}{2R} \right) u_{s} \quad \text{on } \Gamma_{R}.
# $$
#
# The condition is cheap, but approximate: an outgoing circular wave is
# reflected back into $\Omega$ with a relative amplitude $O((kR)^{-2})$
# {cite}`bayliss1982exterior`, and that reflection is one of the two
# leading errors in this demo, the other being the finite element
# discretisation itself. A perfectly matched layer (see the
# {doc}`PML demo <demo_pml>`) absorbs more effectively, at the cost of
# extra layers of cells.
#
# Multiplying by the conjugate of a test function $v$ and integrating by
# parts gives the variational problem (Section 2.3 of
# {cite:t}`ihlenburg1998scattering`): find $u_{s} \in V$ such that
#
# $$
# \int_{\Omega} \left( \nabla u_{s} \cdot \nabla \bar{v} - k^2 u_{s}
# \bar{v} \right) \, {\rm d} x - \int_{\Gamma_{R}} \left(i k -
# \frac{1}{2R} \right) u_{s} \bar{v} \, {\rm d} s = - \int_{\Gamma} i k
# (\boldsymbol{d} \cdot \boldsymbol{n}) u_{i} \bar{v} \, {\rm d} s
# \quad \forall v \in V,
# $$
#
# where $V = H^1(\Omega)$ and $\boldsymbol{n}$ is the outward unit
# normal to $\Omega$ (pointing *into* an obstacle on $\Gamma$).
#
# ## Implementation
#
# We start by importing the necessary modules.

# +
import sys
import time
import typing
from pathlib import Path

from mpi4py import MPI
from petsc4py import PETSc

import gmsh
import numpy as np
import scipy.linalg
import scipy.sparse.linalg
import scipy.spatial
import scipy.special

import ufl
from dolfinx import (
    default_real_type,
    default_scalar_type,
    fem,
    graph,
    has_kahip,
    has_parmetis,
    has_ptscotch,
    io,
    la,
    mesh,
    plot,
)
from dolfinx.fem import petsc as fem_petsc
from dolfinx.fem.petsc import LinearProblem

try:
    import pyvista
except ModuleNotFoundError:
    print("pyvista is required to visualise the solution")
    pyvista = None
# -

# The radiation condition brings a factor $i$ into the problem, so the
# solution and the form above are complex-valued and the form is
# sesquilinear rather than bilinear. DOLFINx can assemble it whatever
# scalar type PETSc was built with, because the scalar type of a form is
# fixed when the form is compiled. The *solver* must match the form: a
# PETSc solve is used when PETSc itself is complex, and SciPy's sparse LU
# factorisation otherwise, which limits the demo to a single process.

# +
# Complex scalar type matching the precision DOLFINx was built with
dtype = np.complex64 if default_real_type == np.float32 else np.complex128
petsc_is_complex = np.issubdtype(default_scalar_type, np.complexfloating)
if not petsc_is_complex and MPI.COMM_WORLD.size > 1:
    PETSc.Sys.Print("PETSc is configured with real scalars: this demo runs in serial only")
    sys.exit(0)
# -

# The parameters of the discrete problem. Resolving a wave needs a fixed
# number of degrees-of-freedom per wavelength, so the cell size is set
# from the wavelength and the polynomial degree. Holding that resolution
# fixed does not hold the error fixed as $k$ grows -- the pollution
# effect {cite}`ihlenburg1995pollution` -- for which raising the degree
# is the cheaper remedy. Away from the obstacles the solution is a
# smooth wave and `h` is adequate; the obstacle boundaries are curved
# and carry the boundary data that drives the scattered field, so cells
# are refined there.

# +
wavelength = 0.5
k = 2 * np.pi / wavelength  # Wavenumber
direction = (1.0, 0.0)  # Propagation direction of the incident wave
degree = 2  # Polynomial degree
nodes_per_wavelength = 12  # Lagrange nodes per wavelength

h = degree * wavelength / nodes_per_wavelength  # Cell size away from obstacles
h_obstacle = h / 2  # Cell size on the obstacle boundaries
# -

# ## Mesh generation
#
# The computational domain is a disc with circular obstacles cut out of
# it. The Gmsh OpenCASCADE kernel builds it with a Boolean cut, after
# which the boundary curves of the result are sorted into the obstacle
# boundary $\Gamma$ and the truncation boundary $\Gamma_{R}$, and tagged
# with named physical groups. A `Distance`/`Threshold` field pair grades
# the cell size from `h_obstacle` on the obstacles to `h` beyond a
# wavelength away from them, and `setOrder(2)` makes the cells curved so
# that the circles are represented to second order.


def generate_mesh(obstacles, radius: float) -> gmsh.model:  # type: ignore[no-any-unimported]
    """Create a Gmsh model of a disc with circular obstacles removed.

    The cell sizes are taken from the parameters set above.

    Args:
        obstacles: Obstacles to remove, each given as a
            ``(x, y, radius)`` tuple.
        radius: Radius of the circular truncation boundary. The
            obstacles must lie strictly inside it.

    Returns:
        Gmsh model holding a generated second-order mesh, with physical
        groups ``"domain"``, ``"obstacle"`` (the obstacle boundaries)
        and ``"radiating"`` (the truncation boundary).
    """
    model = gmsh.model()
    model.add("scattering")
    model.setCurrent("scattering")

    disc = model.occ.addDisk(0.0, 0.0, 0.0, radius, radius)
    holes = [(2, model.occ.addDisk(x, y, 0.0, r, r)) for x, y, r in obstacles]
    (domain,), _ = model.occ.cut([(2, disc)], holes)
    model.occ.synchronize()

    # Sort the boundary curves of the cut domain by evaluating each at
    # the midpoint of its parameter range: a point on the truncation
    # boundary is a distance `radius` from the origin, a point on an
    # obstacle is closer
    obstacle_curves, radiating_curves = [], []
    for dim, tag in model.getBoundary([domain], oriented=False):
        t0, t1 = model.getParametrizationBounds(dim, tag)
        x, y, _ = model.getValue(dim, tag, [(t0[0] + t1[0]) / 2])
        curves = radiating_curves if np.hypot(x, y) > 0.99 * radius else obstacle_curves
        curves.append(tag)

    model.addPhysicalGroup(2, [domain[1]], name="domain")
    model.addPhysicalGroup(1, obstacle_curves, name="obstacle")
    model.addPhysicalGroup(1, radiating_curves, name="radiating")

    model.mesh.field.add("Distance", 1)
    model.mesh.field.setNumbers(1, "CurvesList", obstacle_curves)
    model.mesh.field.add("Threshold", 2)
    model.mesh.field.setNumber(2, "InField", 1)
    model.mesh.field.setNumber(2, "SizeMin", h_obstacle)
    model.mesh.field.setNumber(2, "SizeMax", h)
    model.mesh.field.setNumber(2, "DistMin", 0.0)
    model.mesh.field.setNumber(2, "DistMax", wavelength)
    model.mesh.field.setAsBackgroundMesh(2)

    # Take the cell size from the field alone
    for option in ("FromPoints", "FromCurvature", "ExtendFromBoundary"):
        gmsh.option.setNumber(f"Mesh.MeshSize{option}", 0)

    model.mesh.generate(2)
    model.mesh.setOrder(2)
    return model


# The mesh is built by Gmsh on rank 0 only, and
# {py:func}`model_to_mesh <dolfinx.io.gmsh.model_to_mesh>` distributes it
# over the communicator, returning the mesh together with the tags and a
# map from physical group name to tag.


def create_mesh(obstacles, radius: float) -> io.gmsh.MeshData:
    """Create a distributed mesh of a disc with circular obstacles removed.

    Args:
        obstacles: Obstacles to remove, each given as a
            ``(x, y, radius)`` tuple.
        radius: Radius of the circular truncation boundary.

    The mesh is ghosted by a layer of cells, which the domain
    decomposition solver below uses as the overlap between processes.

    Returns:
        Mesh, its cell and facet tags, and the map from physical group
        name to tag.
    """
    model = generate_mesh(obstacles, radius) if MPI.COMM_WORLD.rank == 0 else None
    return io.gmsh.model_to_mesh(
        model, MPI.COMM_WORLD, rank=0, gdim=2, ghost_mode=mesh.GhostMode.shared_facet
    )


# ## Solver
#
# The incident wave appears both in the variational problem and in the
# post-processing, so it is expressed once as a UFL expression.


def plane_wave(msh, k: float, direction):
    """Incident plane wave of unit amplitude.

    Args:
        msh: Mesh the wave is expressed on.
        k: Wavenumber.
        direction: Unit propagation direction.

    Returns:
        UFL expression for ``exp(i k d.x)``.
    """
    x = ufl.SpatialCoordinate(msh)
    return ufl.exp(1j * k * ufl.dot(ufl.as_vector(direction), x))


# The tagged boundaries carry the two surface terms of the form, and the
# assembled system is factorised by a direct solver. Note the use of
# {py:func}`ufl.inner` rather than {py:func}`ufl.dot` or ``*``: `inner`
# conjugates its second argument, which makes the assembled form the
# sesquilinear $(a, b) = \int_{\Omega} a \bar{b} \, {\rm d}x$. The test
# function must therefore **always** be the second argument.


def scattering_forms(
    mesh_data: io.gmsh.MeshData, radius: float
) -> tuple[ufl.Form, ufl.Form, fem.FunctionSpace]:
    """Build the forms for the field scattered by sound-hard obstacles.

    The wavenumber, incident direction and polynomial degree are taken
    from the parameters set above.

    Args:
        mesh_data: Mesh and tags, as returned by :func:`create_mesh`.
        radius: Radius of the circular truncation boundary.

    Returns:
        Sesquilinear form, linear form, and the space they are posed on.
    """
    msh = mesh_data.mesh
    V = fem.functionspace(msh, ("Lagrange", degree))
    u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
    n = ufl.FacetNormal(msh)
    d = ufl.as_vector(direction)
    ds = ufl.Measure("ds", domain=msh, subdomain_data=mesh_data.facet_tags)
    ds_obstacle = ds(mesh_data.physical_groups["obstacle"].tag)
    ds_radiating = ds(mesh_data.physical_groups["radiating"].tag)

    a = (ufl.inner(ufl.grad(u), ufl.grad(v)) - k**2 * ufl.inner(u, v)) * ufl.dx - (
        1j * k - 1 / (2 * radius)
    ) * ufl.inner(u, v) * ds_radiating
    g = 1j * k * ufl.dot(d, n) * plane_wave(msh, k, direction)
    L = -ufl.inner(g, v) * ds_obstacle
    return a, L, V


def factorisation(comm) -> str:
    """Choose the PETSc package that factorises the operator.

    MUMPS is preferred, then SuperLU_DIST, then the built-in PETSc
    factorisation, which runs on one process only.

    Args:
        comm: MPI communicator the solve runs on.

    Returns:
        Name of the factorisation package.
    """
    petsc_sys = PETSc.Sys()
    if petsc_sys.hasExternalPackage("mumps") and PETSc.IntType != np.int64:
        return "mumps"
    elif petsc_sys.hasExternalPackage("superlu_dist"):
        return "superlu_dist"
    elif comm.size > 1:
        raise RuntimeError("This demo requires a parallel LU solver.")
    else:
        return "petsc"


def solve_scattering(mesh_data: io.gmsh.MeshData, radius: float) -> fem.Function:
    """Solve for the field scattered by sound-hard obstacles.

    Args:
        mesh_data: Mesh and tags, as returned by :func:`create_mesh`.
        radius: Radius of the circular truncation boundary.

    Returns:
        Scattered field.
    """
    msh = mesh_data.mesh
    a, L, V = scattering_forms(mesh_data, radius)
    u_s = fem.Function(V, dtype=dtype, name="u_scattered")
    if petsc_is_complex:
        solver_type = factorisation(msh.comm)
        problem = LinearProblem(
            a,
            L,
            u=u_s,
            petsc_options_prefix="demo_helmholtz_",
            petsc_options={
                "ksp_type": "preonly",
                "pc_type": "lu",
                "pc_factor_mat_solver_type": solver_type,
                "ksp_error_if_not_converged": True,
            },
        )
        problem.solve()
    else:
        # PETSc was built with real scalars and cannot hold the complex
        # operator, so assemble into DOLFINx's own CSR matrix and
        # factorise that with SciPy
        A = fem.assemble_matrix(fem.form(a, dtype=dtype))
        A.scatter_reverse()
        b = fem.assemble_vector(fem.form(L, dtype=dtype))
        b.scatter_reverse(la.InsertMode.add)
        u_s.x.array[:] = scipy.sparse.linalg.spsolve(A.to_scipy(), b.array)

    return u_s


# ## Scattering by a circular cylinder
#
# The first case is a single sound-hard circular cylinder of radius $a$,
# with the domain truncated four wavelengths beyond it. Gmsh is
# initialised here, with its terminal output silenced, and finalised once
# the last mesh has been built.

# +
gmsh.initialize()
gmsh.option.setNumber("General.Terminal", 0)

cylinder_radius = 1.0
radius = cylinder_radius + 4 * wavelength
mesh_data = create_mesh([(0.0, 0.0, cylinder_radius)], radius)
u_s = solve_scattering(mesh_data, radius)
# -

# Scattering of a plane wave $e^{i k x}$ by a sound-hard circular
# cylinder has the classical series solution (Section 2.1.3 of
# {cite:t}`ihlenburg1998scattering`)
#
# $$
# u_{s}(r, \theta) = - \sum_{m=-\infty}^{\infty} i^{m}
# \frac{J_{m}^{\prime}(ka)}{H_{m}^{(1)\prime}(ka)} H_{m}^{(1)}(kr)
# e^{i m \theta},
# $$
#
# with $J_{m}$ the Bessel function of the first kind, $H^{(1)}_{m}$ the
# Hankel function of the first kind, which is the outgoing wave for the
# $e^{-i \omega t}$ convention used here, and a prime denoting
# differentiation with respect to the argument. The series is truncated a
# few terms beyond $ka$, after which the coefficients decay rapidly.


def mie_series(x, k: float, a: float):
    """Field scattered by a sound-hard cylinder, from a series solution.

    The cylinder is centred at the origin and the incident wave is a
    unit-amplitude plane wave travelling in the positive $x$-direction.

    Args:
        x: Points at which to evaluate the field, with shape
            ``(3, num_points)``.
        k: Wavenumber.
        a: Cylinder radius.

    Returns:
        Scattered field at ``x``.
    """
    r, theta = np.hypot(x[0], x[1]), np.arctan2(x[1], x[0])
    num_terms = int(k * a) + 10
    u_s = np.zeros_like(r, dtype=np.complex128)
    for m in range(-num_terms, num_terms + 1):
        c = 1j**m * scipy.special.jvp(m, k * a) / scipy.special.h1vp(m, k * a)
        u_s -= c * scipy.special.hankel1(m, k * r) * np.exp(1j * m * theta)
    return u_s


# The series solution is interpolated into a higher-degree space and
# compared with the computed field. The error combines the finite element
# discretisation error with the error made by the approximate radiation
# condition, and the two are of a similar size here, so neither can be
# removed on its own: halving the cell size alone, or moving $\Gamma_{R}$
# from four to twelve wavelengths beyond the cylinder alone, each leave
# the error at around $2 \times 10^{-2}$, while doing both together
# brings it down to $3 \times 10^{-3}$.

# +
msh = mesh_data.mesh
V_exact = fem.functionspace(msh, ("Lagrange", degree + 2))
u_exact = fem.Function(V_exact, dtype=dtype, name="u_exact")
u_exact.interpolate(lambda x: mie_series(x, k, cylinder_radius).astype(dtype))


def relative_l2_error(u_h: fem.Function, u_ref: fem.Function) -> float:
    """Compute the relative $L^2$ error between two functions.

    Args:
        u_h: Approximate field.
        u_ref: Reference field.

    Returns:
        Relative error.
    """
    comm = u_ref.function_space.mesh.comm
    diff = u_h - u_ref
    error_form = fem.form(ufl.inner(diff, diff) * ufl.dx, dtype=dtype)
    norm_form = fem.form(ufl.inner(u_ref, u_ref) * ufl.dx, dtype=dtype)
    error = comm.allreduce(fem.assemble_scalar(error_form), MPI.SUM)
    norm = comm.allreduce(fem.assemble_scalar(norm_form), MPI.SUM)
    return float(np.sqrt(abs(error) / abs(norm)))


PETSc.Sys.Print(f"Relative L2 error in the scattered field: {relative_l2_error(u_s, u_exact):.2e}")
# -

# ## Focusing by a parabolic reflector
#
# Nothing in the solver is specific to a single obstacle, so the same
# code applies to a geometry for which no series solution is available.
# The reflector below is a chain of overlapping discs centred on the
# parabola $x = -y^2 / (4 f)$, which the Boolean cut fuses into a single
# scalloped obstacle. It is concave towards the incoming wave, so it acts
# as an acoustic mirror that reflects the wave back onto its focus
# $(-f, 0)$.

# +
focal_length = 1.5
aperture = 2.0  # Half-height of the reflector
disc_radius = 0.2  # Larger than half the spacing, so neighbours overlap

y = np.linspace(-aperture, aperture, 13)  # Centres of the discs
reflector = [(-(y_i**2) / (4 * focal_length), y_i, disc_radius) for y_i in y]

radius = 3.5  # Leaves three wavelengths between the focus and the boundary
mesh_data = create_mesh(reflector, radius)
gmsh.finalize()  # No more meshing to do

u_s = solve_scattering(mesh_data, radius)
# -

# The total field $u = u_{i} + u_{s}$ is the quantity of physical
# interest, and is interpolated into the space of the scattered field.

# +
msh = mesh_data.mesh
V = u_s.function_space
u = fem.Function(V, dtype=dtype, name="u_total")
u.interpolate(
    fem.Expression(u_s + plane_wave(msh, k, direction), V.element.interpolation_points, dtype=dtype)
)
# -

# To quantify the focusing, the root-mean-square amplitude over a
# Gaussian window centred on the focus is normalised by the (unit)
# amplitude of the incident wave,
#
# $$
# G = \left( \frac{\int_{\Omega} w |u|^2 \, {\rm d} x}
# {\int_{\Omega} w \, {\rm d} x} \right)^{1/2}, \qquad
# w = e^{-|\boldsymbol{x} - \boldsymbol{x}_{f}|^2 / 2 \sigma^2} .
# $$
#
# The window width $\sigma = \lambda / 2$ is comparable with the
# diffraction-limited spot size $\lambda f / D$ of a reflector of
# aperture $D$. A value greater than one means that the reflector
# concentrates sound at the focus.

# +
sigma = wavelength / 2
x_focus = ufl.as_vector((-focal_length, 0.0))
w = ufl.exp(
    -ufl.dot(ufl.SpatialCoordinate(msh) - x_focus, ufl.SpatialCoordinate(msh) - x_focus)
    / (2 * sigma**2)
)
intensity_form = fem.form(w * ufl.inner(u, u) * ufl.dx, dtype=dtype)
weight_form = fem.form(w * ufl.dx, dtype=dtype)
intensity = msh.comm.allreduce(fem.assemble_scalar(intensity_form), MPI.SUM)
weight = msh.comm.allreduce(fem.assemble_scalar(weight_form), MPI.SUM)
PETSc.Sys.Print(f"Amplitude gain at the focus: {np.sqrt(abs(intensity) / abs(weight)):.2f}")
# -

# The total field is saved in VTKHDF format, for visualisation with,
# e.g., ParaView. A VTKHDF dataset holds one value per mesh node and is
# real-valued, so the real and imaginary parts are written as the two
# components of a vector field, whose magnitude is the amplitude $|u|$.
#
# The mesh geometry is second-order and the solution is of degree two,
# so the degrees-of-freedom of `V` and the nodes of the mesh are the same
# points. Taking the solution cell by cell through the two dofmaps moves
# its values into the node ordering that VTKHDF expects. Only the values
# at nodes owned by this process are written.

# +
results = Path("out_helmholtz")
results.mkdir(exist_ok=True)
filename = results / "u_total.vtkhdf"

u_cells = u.x.array[V.dofmap.list]
values = np.zeros((msh.geometry.x.shape[0], 2), dtype=default_real_type)
geometry_dofmap = msh.geometry.dofmaps[0]
values[geometry_dofmap, 0] = u_cells.real
values[geometry_dofmap, 1] = u_cells.imag

num_nodes_owned = msh.geometry.index_map().size_local
io.vtkhdf.write_mesh(filename, msh)
io.vtkhdf.write_point_data(filename, msh, values[:num_nodes_owned].reshape(-1), 0.0)
# -

# Finally, the real part of the total field is plotted, with the wave
# arriving from the left. Upstream of the reflector the incident and
# reflected fields interfere and the reflected field concentrates near
# the focus, while downstream the reflector casts a shadow. The colour
# scale is clipped at $\pm 2$, so the focal spot saturates it.

if pyvista is not None:
    cells, cell_types, x = plot.vtk_mesh(V)
    grid = pyvista.UnstructuredGrid(cells, cell_types, x)
    grid.point_data["u"] = u.x.array.real
    plotter = pyvista.Plotter()
    plotter.add_mesh(grid, cmap="RdBu_r", clim=(-2, 2), show_scalar_bar=True)
    plotter.view_xy()
    if pyvista.OFF_SCREEN:
        plotter.screenshot(results / "u_total.png", window_size=[800, 800])
    else:
        plotter.show()

# ## Solving without a direct factorisation
#
# Both problems above were factorised directly, which is the right
# choice at this size but does not scale: the factors of a
# three-dimensional Helmholtz problem fill in quickly. The alternative
# is a Krylov method, and here the Helmholtz operator is awkward. It is
# indefinite, so neither multigrid nor an incomplete factorisation
# preconditions it robustly {cite}`gander2019helmholtz`, and
# unpreconditioned GMRES needs some 3000 iterations on the problem
# solved above.
#
# Overlapping Schwarz domain decomposition is the standard answer
# {cite}`dolean2015ddm`. The domain is covered by overlapping
# subdomains $\Omega_i$ and the preconditioner solves a Helmholtz
# problem on each of them,
#
# $$
# M^{-1} = \sum_i R_i^{T} D_i A_i^{-1} R_i,
# $$
#
# where $R_i$ restricts a vector to the degrees-of-freedom of
# $\Omega_i$ and the diagonal $D_i$ is a partition of unity, so that
# $\sum_i R_i^{T} D_i R_i = I$. Two choices decide whether this works.
#
# **The transmission condition** is what each local problem imposes on
# the artificial interface $\Gamma_i$ where it was cut out of the
# domain. The obvious choice, zero Dirichlet data, is a poor one for the
# Helmholtz equation: the local problem is close to singular whenever
# $k^2$ is near a Dirichlet eigenvalue of a subdomain, and a wave
# leaving a subdomain is reflected at the interface instead of passing
# through. Replacing the impedance condition below by zero Dirichlet
# data costs two orders of magnitude in iterations. Imposing instead
# the impedance condition of Després {cite}`despres1990helmholtz`,
#
# $$
# \frac{\partial u}{\partial n} - i k u = 0 \quad
# \text{on } \Gamma_i,
# $$
#
# makes every local problem well posed and lets a wave leave. This is
# the optimised restricted additive Schwarz (ORAS) method
# {cite}`stcyr2007oras`, and it is the same condition used on the
# truncation boundary, applied to an internal interface.
#
# **The coarse space.** A one-level method couples neighbours only, so
# information crosses one subdomain per iteration and the iteration
# count grows with the number of subdomains. A coarse space fixes this
# by adding a second, global solve on a small space. For the Helmholtz
# equation a good choice is built from the Dirichlet-to-Neumann (DtN)
# operator {cite}`nataf2011dtn,conen2014dtn`: on each subdomain,
# discretised, it is the Schur complement of the local operator on the
# interface,
#
# $$
# S_i = A_{\Gamma \Gamma} - A_{\Gamma I} A_{II}^{-1} A_{I \Gamma},
# $$
#
# and the eigenvectors of $S_i g = \lambda M_{\Gamma} g$, with
# $M_{\Gamma}$ the mass matrix of the interface, with the smallest
# $|\lambda|$ are the interface traces that the local solves damp
# least. Each is extended into the subdomain as a solution of the
# Helmholtz equation, weighted by the partition of unity, and the
# results are the columns of $Z$. The correction is applied by
# deflation,
#
# $$
# M_{2}^{-1} = Q + M^{-1} (I - A Q), \qquad Q = Z E^{-1} Z^{*},
# \qquad E = Z^{*} A Z .
# $$
#
# ## Subdomains in parallel
#
# The mesh is already distributed, and that distribution is the
# decomposition: each process splits the cells it holds into a few
# subdomains and builds each one as a mesh on {py:data}`MPI.COMM_SELF
# <mpi4py.MPI.COMM_SELF>`, on which the local operator is assembled
# and factorised like any other problem. The overlap between processes
# is the layer of ghost cells, so the mesh is created with
# {py:attr}`GhostMode.shared_facet <dolfinx.mesh.GhostMode>`.
#
# The two parallel operations in $R_i^{T} D_i A_i^{-1} R_i$ are the
# ones a distributed vector already provides: $R_i$ needs the values on
# the ghost degrees-of-freedom, which is a forward scatter, and
# $R_i^{T} D_i$ keeps the entries this process owns. Nothing else in
# the Schwarz operator communicates. The coarse space is distributed in
# the same way, each process holding the columns of $Z$ belonging to
# its own subdomains, and only the small matrix $E$ and the coarse
# right-hand side are reduced across processes.
#
# Only the operator and the Krylov method depend on the linear algebra
# backend. With complex PETSc scalars the matrix is a PETSc one, the
# preconditioner is given to PETSc as a shell, and the solve runs on
# any number of processes. On a single process the problem also fits
# DOLFINx's own matrix and SciPy's GMRES, which keeps the method
# available in a build whose PETSc has real scalars. Where both apply
# the demo runs both; everything between them -- subdomains, local
# solves, coarse space -- is the same code, and the two solutions agree
# with the direct one.
#
# The two iteration counts differ a little because the stopping tests
# do: SciPy measures the true residual against the right-hand side,
# while PETSc measures the preconditioned residual, which is the
# quantity its left-preconditioned GMRES minimises.

# +
if not petsc_is_complex and msh.comm.size > 1:
    PETSc.Sys.Print("Without complex PETSc scalars the solver below runs in serial only")
    sys.exit(0)
if not (has_parmetis or has_ptscotch or has_kahip):
    PETSc.Sys.Print("The solver below needs DOLFINx built with a graph partitioner")
    sys.exit(0)

subdomains_per_process = 4
overlap = 1  # Cell layers; across processes it is the ghost layer
coarse_threshold = 2 * k  # Keep DtN modes with |lambda| below this
iteration_limit = 500  # Cap, so that a method that stalls does not run away
restart = 100  # GMRES restart length

# Subdomain data is matched to the distributed mesh by position. The
# tolerance sits well inside a cell, and well above the error of the
# single-precision coordinate arithmetic of a float32 build, where the
# two meshes compute the same point through different cells.
match_tolerance = 1e-3 * h_obstacle
tolerance = np.sqrt(np.finfo(dtype).eps)  # Reachable in single and double precision
# -


class Subdomain(typing.NamedTuple):
    """Operators and index sets for one overlapping subdomain.

    Args:
        dofs: For each degree-of-freedom of the subdomain, the index of
            the matching degree-of-freedom of the distributed space on
            this process.
        weight: Degrees-of-freedom this subdomain is responsible for,
            which is the partition of unity of the Schwarz operator.
        share: Partition of unity of the coarse space, which shares a
            degree-of-freedom equally between the subdomains holding it.
        interface: Positions in ``dofs`` of the degrees-of-freedom on
            the artificial interface.
        interior: Positions in ``dofs`` of the remaining
            degrees-of-freedom.
        neumann: Helmholtz operator of the subdomain, which leaves a
            natural condition on the artificial interface.
        mass: Mass matrix of the artificial interface.
        impedance: Factorised local operator, with the impedance
            condition on the artificial interface.
        interior_lu: Factorised interior block of ``neumann``, which
            gives the Helmholtz-harmonic extension of interface data
            used by the coarse space.
    """

    dofs: np.ndarray
    weight: np.ndarray
    share: np.ndarray
    interface: np.ndarray
    interior: np.ndarray
    neumann: typing.Any
    mass: typing.Any
    impedance: typing.Any
    interior_lu: typing.Any


def partition_local_cells(msh, num_parts: int, layers: int):
    """Split the cells held by this process into overlapping parts.

    The cells are partitioned with {py:func}`partition
    <dolfinx.graph.partition>` on ``MPI.COMM_SELF``, so each process
    decides its own subdomains without communicating, and each part is
    then grown by whole layers of cells.

    Args:
        msh: Distributed mesh.
        num_parts: Number of subdomains for this process.
        layers: Number of cell layers each part is grown by.

    Returns:
        Cells of each grown subdomain.
    """
    tdim = msh.topology.dim
    cell_map = msh.topology.index_map(tdim)
    local = np.arange(cell_map.size_local + cell_map.num_ghosts, dtype=np.int32)
    c_to_v = msh.topology.connectivity(tdim, 0)
    vertices = msh.topology.index_map(0).local_to_global(c_to_v.array)
    cells = graph.adjacencylist(vertices.reshape(c_to_v.num_nodes, -1)[local])
    dual = mesh.build_dual_graph(MPI.COMM_SELF, msh.topology.cell_type, cells)
    destinations = graph.partition(MPI.COMM_SELF, num_parts, dual)
    part = np.fromiter((destinations.links(i)[0] for i in local), dtype=np.int32, count=local.size)

    partition = []
    for i in range(num_parts):
        grown = local[part == i]
        for _ in range(layers):
            incident = mesh.compute_incident_entities(msh.topology, grown, tdim, 0)
            grown = mesh.compute_incident_entities(msh.topology, incident, 0, tdim)
        partition.append(grown)
    return partition


def subdomain_mesh(msh, cells):
    """Build a mesh of ``cells`` on this process alone.

    Args:
        msh: Mesh the cells belong to.
        cells: Cells of the subdomain.

    Returns:
        Mesh of those cells, on ``MPI.COMM_SELF``.
    """
    geometry_dofmap = msh.geometry.dofmaps[0][cells]
    nodes = np.unique(geometry_dofmap)
    renumber = np.full(msh.geometry.x.shape[0], -1, dtype=np.int64)
    renumber[nodes] = np.arange(nodes.size)
    # A domain of its own, so that UFL can tell forms on the subdomain
    # apart from forms on the distributed mesh
    domain = ufl.Mesh(msh.ufl_domain().ufl_coordinate_element())
    # Node coordinates are always stored with three components, so the
    # unused ones are dropped: create_mesh takes the geometric dimension
    # from the width of the array it is given
    return mesh.create_mesh(
        MPI.COMM_SELF,
        renumber[geometry_dofmap].astype(np.int64),
        domain,
        msh.geometry.x[nodes, : msh.geometry.dim],
    )


def classify_boundary(sub_msh, mesh_data: io.gmsh.MeshData, tags):
    """Sort the boundary facets of a subdomain by what they sit on.

    A facet of the subdomain boundary either lies on the boundary of
    the full mesh, where the physical condition applies, or cuts
    through it, where the transmission condition does. They are told
    apart by matching facet midpoints against the tagged facets.

    Args:
        sub_msh: Subdomain mesh.
        mesh_data: Mesh and tags of the full mesh.
        tags: Physical group tags to match against.

    Returns:
        Facets on each of the given tags, and those on the artificial
        interface under the key ``"interface"``.
    """
    fdim = sub_msh.topology.dim - 1
    sub_msh.topology.create_connectivity(fdim, fdim + 1)
    boundary = mesh.exterior_facet_indices(sub_msh.topology)
    midpoints = mesh.compute_midpoints(sub_msh, fdim, boundary)

    assert mesh_data.facet_tags is not None, "Facet tags are missing"
    facets = {}
    matched = np.zeros(boundary.size, dtype=bool)
    for name, tag in tags.items():
        tagged = mesh_data.facet_tags.find(tag)
        if tagged.size > 0:
            tree = scipy.spatial.cKDTree(mesh.compute_midpoints(mesh_data.mesh, fdim, tagged))
            on_tag = tree.query(midpoints)[0] < match_tolerance
        else:
            on_tag = np.zeros(boundary.size, dtype=bool)
        facets[name] = boundary[on_tag]
        matched |= on_tag
    facets["interface"] = boundary[~matched]
    return facets


def build_subdomains(mesh_data: io.gmsh.MeshData, V: fem.FunctionSpace, radius: float):
    """Assemble the local operators of an overlapping decomposition.

    The number of subdomains per process and the overlap are taken from
    the parameters set above.

    Args:
        mesh_data: Mesh and tags, as returned by :func:`create_mesh`.
        V: Space the distributed problem is posed on.
        radius: Radius of the circular truncation boundary.

    Returns:
        List of subdomains held by this process.
    """
    msh = mesh_data.mesh
    tdim = msh.topology.dim
    tags = {
        "obstacle": mesh_data.physical_groups["obstacle"].tag,
        "radiating": mesh_data.physical_groups["radiating"].tag,
    }
    x_dofs = scipy.spatial.cKDTree(V.tabulate_dof_coordinates())
    index_map = V.dofmap.index_map
    num_local = index_map.size_local + index_map.num_ghosts

    built = []
    for cells in partition_local_cells(msh, subdomains_per_process, overlap):
        sub_msh = subdomain_mesh(msh, cells)
        facets = classify_boundary(sub_msh, mesh_data, tags)

        # On a subdomain mesh the artificial interface is part of the
        # boundary, so the local forms need no special integration
        # entities
        marked = np.concatenate([facets["radiating"], facets["interface"]])
        values = np.concatenate(
            [
                np.full(facets["radiating"].size, 1),
                np.full(facets["interface"].size, 2),
            ]
        ).astype(np.int32)
        order = np.argsort(marked)
        facet_tags = mesh.meshtags(sub_msh, tdim - 1, marked[order].astype(np.int32), values[order])

        V_sub = fem.functionspace(sub_msh, ("Lagrange", degree))
        u, v = ufl.TrialFunction(V_sub), ufl.TestFunction(V_sub)
        ds_sub = ufl.Measure("ds", domain=sub_msh, subdomain_data=facet_tags)
        a = (ufl.inner(ufl.grad(u), ufl.grad(v)) - k**2 * ufl.inner(u, v)) * ufl.dx - (
            1j * k - 1 / (2 * radius)
        ) * ufl.inner(u, v) * ds_sub(1)
        m = ufl.inner(u, v) * ds_sub(2)
        A = fem.assemble_matrix(fem.form(a, dtype=dtype)).to_scipy().tocsc()
        M = fem.assemble_matrix(fem.form(m, dtype=dtype)).to_scipy().tocsc()

        # The subdomain's degrees-of-freedom are a subset of those this
        # process holds, matched by position
        distance, dofs = x_dofs.query(V_sub.tabulate_dof_coordinates())
        assert distance.max() < match_tolerance, "subdomain degrees-of-freedom not matched"
        interface = np.sort(fem.locate_dofs_topological(V_sub, tdim - 1, facets["interface"]))
        interior = np.setdiff1d(np.arange(dofs.size), interface)
        built.append((dofs.astype(np.int32), interface, interior, A, M))

    # Partition of unity. A degree-of-freedom is the responsibility of
    # the process owning it and, there, of the first subdomain holding
    # it; it is shared with every subdomain that overlaps it, counted
    # over all processes.
    counter = la.vector(index_map, 1, dtype=dtype)
    counter.array[:] = 0
    for dofs, *_ in built:
        counter.array[dofs] += 1
    counter.scatter_reverse(la.InsertMode.add)
    counter.scatter_forward()
    multiplicity = counter.array.real

    claimed = np.zeros(num_local, dtype=bool)
    subdomains = []
    for dofs, interface, interior, A, M in built:
        weight = (dofs < index_map.size_local) & ~claimed[dofs]
        claimed[dofs[weight]] = True
        subdomains.append(
            Subdomain(
                dofs=dofs,
                weight=weight,
                share=1.0 / multiplicity[dofs],
                interface=interface,
                interior=interior,
                neumann=A,
                mass=M,
                impedance=scipy.sparse.linalg.splu((A - 1j * k * M).tocsc()),
                interior_lu=scipy.sparse.linalg.splu(A[interior][:, interior].tocsc()),
            )
        )
    return subdomains


# The Schwarz operator itself: a forward scatter, a local solve on each
# subdomain, and the entries this process is responsible for.


def schwarz(subdomains, scatter, num_owned: int):
    """Build the one-level restricted additive Schwarz preconditioner.

    Args:
        subdomains: Local operators held by this process.
        scatter: Function filling the ghost entries of a vector.
        num_owned: Number of degrees-of-freedom owned by this process.

    Returns:
        Function applying the preconditioner to a residual.
    """

    def apply(r_owned):
        r = scatter(r_owned)
        z = np.zeros(num_owned, dtype=dtype)
        for s in subdomains:
            local = s.impedance.solve(r[s.dofs])
            z[s.dofs[s.weight]] += local[s.weight]
        return z

    return apply


def dtn_coarse_space(subdomains, num_owned: int, threshold: float):
    """Build a coarse space from the Dirichlet-to-Neumann operator.

    On each subdomain the discrete DtN operator on the artificial
    interface is the Schur complement of the local operator,
    ``S = A_GG - A_GI A_II^-1 A_IG``. The eigenvectors of
    ``S g = lambda M g`` with the smallest eigenvalues are the traces
    the local solves damp least. Each is extended into the subdomain as
    a solution of the Helmholtz equation and weighted by the partition
    of unity.

    Args:
        subdomains: Local operators held by this process.
        num_owned: Number of degrees-of-freedom owned by this process.
        threshold: Keep the modes whose eigenvalue satisfies
            ``|lambda| < threshold``.

    Returns:
        Columns of the coarse space belonging to this process.
    """
    columns = []
    for s in subdomains:
        extension = -s.interior_lu.solve(s.neumann[s.interior][:, s.interface].toarray())
        S = (
            s.neumann[s.interface][:, s.interface].toarray()
            + s.neumann[s.interface][:, s.interior] @ extension
        )
        mass = s.mass[s.interface][:, s.interface].toarray()
        eigenvalues, modes = scipy.linalg.eig(S, mass)
        for j in np.flatnonzero(np.abs(eigenvalues) < threshold):
            local = np.zeros(s.dofs.size, dtype=dtype)
            local[s.interface] = modes[:, j]
            local[s.interior] = extension @ modes[:, j]
            column = np.zeros(num_owned, dtype=dtype)
            column[s.dofs[s.weight]] = (s.share * local)[s.weight]
            columns.append(column)
    return columns


def two_level(columns, matvec, one_level, num_owned: int, comm):
    """Add a coarse correction to a one-level preconditioner.

    The coarse operator ``E = Z* A Z`` is assembled with one mat-vec
    per coarse mode and reduced across processes, then factorised
    redundantly: it is small enough that every process holding a copy
    is cheaper than solving it in parallel. The correction is applied
    by deflation.

    Args:
        columns: Coarse space columns belonging to this process.
        matvec: Function applying the operator.
        one_level: One-level preconditioner.
        num_owned: Number of degrees-of-freedom owned by this process.
        comm: MPI communicator.

    Returns:
        Function applying the two-level preconditioner, and the size of
        the coarse space.
    """
    per_process = comm.allgather(len(columns))
    offset = int(np.sum(per_process[: comm.rank]))
    size = int(np.sum(per_process))

    E = np.zeros((size, size), dtype=dtype)
    empty = np.zeros(num_owned, dtype=dtype)
    for rank in range(comm.size):
        start = int(np.sum(per_process[:rank]))
        for j in range(per_process[rank]):
            # Every process takes part in the mat-vec and holds the
            # rows of the result that it owns
            z = columns[j] if rank == comm.rank else empty
            w = matvec(z)
            for i, column in enumerate(columns):
                E[offset + i, start + j] = np.vdot(column, w)
    E = comm.allreduce(E, MPI.SUM)
    factorised = scipy.linalg.lu_factor(E)

    def coarse(r_owned):
        rhs = np.zeros(size, dtype=dtype)
        for i, column in enumerate(columns):
            rhs[offset + i] = np.vdot(column, r_owned)
        y = scipy.linalg.lu_solve(factorised, comm.allreduce(rhs, MPI.SUM))
        out = np.zeros_like(r_owned)
        for i, column in enumerate(columns):
            out += y[offset + i] * column
        return out

    def apply(r_owned):
        z = coarse(r_owned)
        return z + one_level(r_owned - matvec(z))

    return apply, size


class SchwarzPreconditioner:
    """Apply a Schwarz preconditioner to a PETSc vector."""

    def __init__(self, apply):
        """Wrap a function acting on the owned entries of a vector.

        Args:
            apply: Function applying the preconditioner.
        """
        self._apply = apply

    def apply(self, pc, x, y):  # noqa: D102 (PETSc interface)
        y.array_w[:] = self._apply(x.array_r)


# The problem solved above is assembled again and solved with each
# preconditioner, with the direct solution as the reference.


def petsc_backend(a: ufl.Form, L: ufl.Form, comm):
    """Assemble with PETSc and solve with its GMRES.

    Args:
        a: Sesquilinear form.
        L: Linear form.
        comm: MPI communicator.

    Returns:
        Functions applying the operator, solving the system with a
        given preconditioner, and solving it by factorisation.
    """
    A = fem_petsc.assemble_matrix(fem.form(a, dtype=dtype))
    A.assemble()
    rhs = fem_petsc.assemble_vector(fem.form(L, dtype=dtype))
    rhs.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    x_vec, y_vec = A.createVecRight(), A.createVecLeft()

    def matvec(x_owned):
        x_vec.array_w[:] = x_owned
        A.mult(x_vec, y_vec)
        return y_vec.array_r.copy()

    def solve(preconditioner):
        x = rhs.duplicate()
        ksp = PETSc.KSP().create(comm)  # type: ignore[arg-type]
        ksp.setOperators(A)
        ksp.setType("gmres")
        ksp.setGMRESRestart(restart)
        ksp.setTolerances(rtol=tolerance, max_it=iteration_limit)
        ksp.getPC().setType("python")
        ksp.getPC().setPythonContext(SchwarzPreconditioner(preconditioner))
        ksp.solve(rhs, x)
        solution = x.array_r.copy()
        iterations, converged = ksp.getIterationNumber(), ksp.is_converged
        ksp.destroy()
        x.destroy()
        return solution, iterations, converged

    def direct():
        x = rhs.duplicate()
        ksp = PETSc.KSP().create(comm)  # type: ignore[arg-type]
        ksp.setOperators(A)
        ksp.setType("preonly")
        ksp.getPC().setType("lu")
        ksp.getPC().setFactorSolverType(factorisation(comm))
        ksp.solve(rhs, x)
        solution = x.array_r.copy()
        ksp.destroy()
        x.destroy()
        return solution

    return matvec, solve, direct


def scipy_backend(a: ufl.Form, L: ufl.Form, comm):
    """Assemble into DOLFINx's matrix and solve with SciPy's GMRES.

    On one process the matrix has no ghost rows or columns, so it maps
    directly onto a SciPy sparse matrix.

    Args:
        a: Sesquilinear form.
        L: Linear form.
        comm: MPI communicator, which must hold one process.

    Returns:
        Functions applying the operator, solving the system with a
        given preconditioner, and solving it by factorisation.
    """
    assert comm.size == 1, "the SciPy backend runs on one process"
    A = fem.assemble_matrix(fem.form(a, dtype=dtype))
    A.scatter_reverse()
    rhs = fem.assemble_vector(fem.form(L, dtype=dtype))
    rhs.scatter_reverse(la.InsertMode.add)
    operator = A.to_scipy()

    def matvec(x_owned):
        return operator @ x_owned

    def solve(preconditioner):
        iterations = 0

        def count(_):
            nonlocal iterations
            iterations += 1

        x, info = scipy.sparse.linalg.gmres(
            operator,
            rhs.array,
            M=scipy.sparse.linalg.LinearOperator(
                operator.shape, preconditioner, dtype=operator.dtype
            ),
            restart=restart,
            maxiter=max(iteration_limit // restart, 1),
            rtol=tolerance,
            callback=count,
            callback_type="pr_norm",
        )
        return x, iterations, info == 0

    def direct():
        return scipy.sparse.linalg.spsolve(operator.tocsc(), rhs.array)

    return matvec, solve, direct


# +
tdim = msh.topology.dim
for pair in ((tdim, 0), (0, tdim), (tdim - 1, tdim), (tdim, tdim - 1)):
    msh.topology.create_connectivity(*pair)

a, L, V = scattering_forms(mesh_data, radius)
index_map = V.dofmap.index_map
num_owned = index_map.size_local
ghosted = la.vector(index_map, 1, dtype=dtype)


def scatter(x_owned):
    """Owned values of a vector, with its ghost entries filled."""
    ghosted.array[:num_owned] = x_owned
    ghosted.scatter_forward()
    return ghosted.array


# PETSc needs complex scalars to hold this operator, and SciPy needs
# the problem to sit on one process; where both hold, both are run.
backends = []
if petsc_is_complex:
    backends.append(("PETSc", petsc_backend(a, L, msh.comm)))
if msh.comm.size == 1:
    backends.append(("SciPy", scipy_backend(a, L, msh.comm)))
matvec = backends[0][1][0]

# Times are the slowest process, which is what a user waits for
start = time.perf_counter()
subdomains = build_subdomains(mesh_data, V, radius)
one_level = schwarz(subdomains, scatter, num_owned)
subdomain_time = msh.comm.allreduce(time.perf_counter() - start, MPI.MAX)

start = time.perf_counter()
columns = dtn_coarse_space(subdomains, num_owned, coarse_threshold)
two_level_apply, coarse_size = two_level(columns, matvec, one_level, num_owned, msh.comm)
coarse_time = msh.comm.allreduce(time.perf_counter() - start, MPI.MAX)

reference = u_s.x.array[:num_owned]
norm = np.sqrt(msh.comm.allreduce(float(np.vdot(reference, reference).real), MPI.SUM))
num_subdomains = msh.comm.allreduce(len(subdomains), MPI.SUM)
PETSc.Sys.Print(
    f"\n{num_subdomains} subdomains over {msh.comm.size} process(es), "
    f"{coarse_size} coarse modes, {index_map.size_global} degrees-of-freedom"
)
PETSc.Sys.Print(f"setup: subdomains {subdomain_time:.2f} s, coarse space {coarse_time:.2f} s")
PETSc.Sys.Print(f"{'solver':<24}{'backend':>8}{'iterations':>12}{'solve (s)':>11}{'error':>10}")


def report(label: str, name: str, solution, iterations, elapsed: float):
    """Print one row of the comparison.

    Args:
        label: Name of the solver.
        name: Name of the linear algebra backend.
        solution: Degrees-of-freedom owned by this process.
        iterations: Iteration count, or ``None`` for a direct solve.
        elapsed: Wall time of the solve.
    """
    difference = solution - reference
    error = np.sqrt(msh.comm.allreduce(float(np.vdot(difference, difference).real), MPI.SUM))
    count = "--" if iterations is None else str(iterations)
    PETSc.Sys.Print(f"{label:<24}{name:>8}{count:>12}{elapsed:>11.2f}{error / norm:>10.1e}")


# The factorisation the demo used above, timed here for comparison: it
# is the cost the iterative solvers have to beat
for name, (_, _, direct) in backends:
    start = time.perf_counter()
    solution = direct()
    elapsed = msh.comm.allreduce(time.perf_counter() - start, MPI.MAX)
    report("direct (LU)", name, solution, None, elapsed)

for label, preconditioner in (
    ("one-level (ORAS)", one_level),
    ("two-level (ORAS + DtN)", two_level_apply),
):
    for name, (_, solve, _) in backends:
        start = time.perf_counter()
        solution, iterations, converged = solve(preconditioner)
        elapsed = msh.comm.allreduce(time.perf_counter() - start, MPI.MAX)
        report(label, name, solution, iterations if converged else None, elapsed)
# -

# The direct solve is the fastest of the three, which is what the top
# of this section said to expect: at thirty thousand
# degrees-of-freedom in two dimensions a sparse factorisation is cheap,
# and the iterative methods have not even recovered their setup cost.
# They are not competing with a factorisation on this problem. They are
# what is left when the factors no longer fit, which a
# three-dimensional problem reaches quickly. What transfers from the
# table is therefore the iteration counts and how they grow, not the
# times.
#
# The one-level method converges in a few tens of iterations against
# the few thousand an unpreconditioned solve needs, and its iteration
# count grows with the number of subdomains, because information
# crosses one subdomain per iteration. Limiting that growth is what the
# coarse space is for, and it does, by about a quarter here:
#
# | processes | subdomains | one-level | two-level |
# | --------- | ---------- | --------- | --------- |
# | 1         | 4          | 47        | 34        |
# | 2         | 8          | 78        | 53        |
# | 4         | 16         | 99        | 80        |
#
# Fewer iterations is not the same as less time, and on this problem
# the two-level method is the slower of the two. Every one of its
# iterations carries a coarse solve and the extra mat-vec of the
# deflation, and the coarse space has to be built first, which costs
# more than a second against a solve of well under one.
#
# That balance, rather than the iteration count, is what decides
# whether a coarse space is worth having. It turns in its favour when
# iterations are expensive relative to the coarse solve -- many more
# subdomains, or a problem the one-level method handles badly -- and
# when the same coarse space is reused, since building it is a one-off
# while the iterations are paid on every right-hand side. The overlap
# matters too: with two layers instead of one, and subdomains cut
# without regard to the process boundaries, the one-level method needs
# 43 iterations at 16 subdomains and the coarse space does not repay
# even its iteration cost.
#
# The transmission condition matters more than either, and by a wider
# margin than any of this. Measured on the sixteen-subdomain
# decomposition described above, zero Dirichlet data in place of the
# impedance condition needs 8800 iterations where the impedance
# condition needs 43; on a decomposition four times finer it does not
# converge at all, while adding the coarse space to it brings it back
# to 388. A coarse space is worth most where the one-level method is
# weakest.

# ## References
# ```{bibliography}
# :filter: docname in docnames
# ```
