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
# {cite}`bayliss1982exterior`, and that reflection is the dominant error
# in this demo. A perfectly matched layer (see the
# {doc}`PML demo <demo_pml>`) absorbs more effectively, at the cost of
# extra layers of cells.
#
# Multiplying by the conjugate of a test function $v$ and integrating by
# parts gives the variational problem (see, e.g.,
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
from pathlib import Path

from mpi4py import MPI
from petsc4py import PETSc

import gmsh
import numpy as np
import scipy.sparse.linalg
import scipy.special

import ufl
from dolfinx import default_real_type, default_scalar_type, fem, io, la, plot
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

    Returns:
        Mesh, its cell and facet tags, and the map from physical group
        name to tag.
    """
    model = generate_mesh(obstacles, radius) if MPI.COMM_WORLD.rank == 0 else None
    return io.gmsh.model_to_mesh(model, MPI.COMM_WORLD, rank=0, gdim=2)


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


def solve_scattering(mesh_data: io.gmsh.MeshData, radius: float) -> fem.Function:
    """Solve for the field scattered by sound-hard obstacles.

    The wavenumber, incident direction and polynomial degree are taken
    from the parameters set above.

    Args:
        mesh_data: Mesh and tags, as returned by :func:`create_mesh`.
        radius: Radius of the circular truncation boundary.

    Returns:
        Scattered field.
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

    u_s = fem.Function(V, dtype=dtype, name="u_scattered")
    if petsc_is_complex:
        # Prefer MUMPS for the factorisation (not available for 64-bit
        # indices), then SuperLU_DIST, then the built-in (serial) PETSc
        # factorisation
        petsc_sys = PETSc.Sys()
        if petsc_sys.hasExternalPackage("mumps") and PETSc.IntType != np.int64:
            solver_type = "mumps"
        elif petsc_sys.hasExternalPackage("superlu_dist"):
            solver_type = "superlu_dist"
        elif msh.comm.size > 1:
            raise RuntimeError("This demo requires a parallel LU solver.")
        else:
            solver_type = "petsc"

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
# cylinder has the classical series solution (Chapter 2 of
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
# condition, the second dominating here: refining the mesh barely moves
# the number, whereas moving $\Gamma_{R}$ further out reduces it.

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

# ## References
# ```{bibliography}
# :filter: docname in docnames
# ```
