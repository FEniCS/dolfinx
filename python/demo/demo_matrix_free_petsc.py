# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.13.6
# ---

# # Matrix-free solvers in DOLFINx using PETSc
#
# Author: Jørgen S. Dokken
#
# This demo can be downloaded as a single Python file
# {download}`demo_matrix_free_petsc.py`.
# In this demo, we will demonstrate how to set up a fully matrix-free
# solver using PETSc.
# We will start by defining our variational problem, and then in turn
# define a custom PETSc-matrix that will handle assembly without ever
# forming the full system matrix.

# ## Problem definition
# In this example, we consider a projection problem, i.e.
# Find $(u_h, p_h) \in V_h \times Q_h$ such that
#
# $$
# \begin{align}
#   \min_{u_h, p_h} J(u_h, p_h) &=
#   \frac{1}{2} \int_\Omega \vert u_h - f\vert^2~\mathrm{d}x
#   + \int_\Omega \vert p_h - g\vert^2~\mathrm{d}x
# \end{align}
# $$
#
# By considering the optimality conditions of this system we arrive at the
# variational problem:
# Find $(u_h, p_h) \in V_h \times Q_h$ such that
#
# $$
# \begin{align}
#   \int_\Omega (u_h-f) \cdot v~\mathrm{d}x + \int_\Omega (p_h-g) q
#  ~\mathrm{d}x
# &= 0 \quad \forall (v, q) \in V_h \times Q_h
# \end{align}
# $$
#
# We start by importing the necessary modules

# +
from mpi4py import MPI
from petsc4py import PETSc

import numpy as np

import basix.ufl
import dolfinx.fem.petsc
import ufl

# -

# ## Matrix-free operator
# Many interative methods, such as the conjugate gradient method,
# only requires the action of the system matrix on a vector.
# Thus, one can assemble a form of rank 1 with a given function replacing
# the trial function and obtain this vector.

# We do this using `ufl.action` to create a rank 1 form.
# Additionally, some preconditioners, such as the Jacobi preconditioner,
# need access to the diagonal of the matrix.
# This can be obtained by assembling the form with a special option to only
# compute the diagonal, i.e.
# the `form_compiler_options={"part":"diagonal"}`, which is passed to FFCx
# when calling `dolfinx.fem.form`.

# For the assembly itself, we require an operator that can compute the
# action of the matrix on a vector, as well as provide the diagonal.
# We provide this class below:


class MatrixFreeOperator:
    # Data allocation for the operator
    _w: dolfinx.fem.Function | list[dolfinx.fem.Function]  # Store working solution
    _diagonal: PETSc.Vec  # Temporary storage of diagonal
    _vector: PETSc.Vec  # Temporary storage of action

    _vector_product: dolfinx.fem.Form | list[dolfinx.fem.Form]  # Compiled matrix-vector product
    _compiled_diagonal: dolfinx.fem.Form | list[ufl.form.Form]  # Compiled diagonal form

    def __init__(
        self,
        bilinear_form: ufl.Form,
        bcs: list[dolfinx.fem.DirichletBC] | None = None,
        form_compiler_options: dict | None = None,
        jit_options: dict | None = None,
    ):
        """A matrix-free operator for a bilinear form with
        boundary conditions.

        Args:
            bilinear_form: The bilinear form.
            bcs: A list of Dirichlet boundary conditions.
        """
        jit_options = {} if jit_options is None else jit_options
        form_compiler_options = {} if form_compiler_options is None else form_compiler_options
        diagnal_options = form_compiler_options.copy()
        diagnal_options["part"] = "diagonal"

        # Store the boundary conditions
        self._bcs = [] if bcs is None else bcs

        # Use the number of arguments in the bilinear for to decide if we
        # have a mixed function space
        arguments = bilinear_form.arguments()
        if len(arguments) > 2:
            # Handle MixedFunctionSpace forms
            size = max(arg.part() for arg in arguments) + 1
            assert max(arg.number() for arg in arguments) == 1
            a_blocked = ufl.extract_blocks(bilinear_form)
            assert len(a_blocked) == size
            spaces = [a_blocked[i][i].arguments()[0].ufl_function_space() for i in range(size)]

            self._w = [dolfinx.fem.Function(space) for space in spaces]
            self._diagonal = dolfinx.fem.petsc.create_vector(spaces)
            self._vector = dolfinx.fem.petsc.create_vector(spaces)
            self._vector_product = dolfinx.fem.form(
                ufl.extract_blocks(ufl.action(bilinear_form, self._w)),
                form_compiler_options=form_compiler_options,
                jit_options=jit_options,
            )
            self._compiled_diagonal = dolfinx.fem.form(
                [a_blocked[i][i] for i in range(size)],
                form_compiler_options=diagnal_options,
                jit_options=jit_options,
            )
        else:
            # Handle "standard" bilinear forms
            assert len(arguments) == 2, "Only bilinear forms are supported"
            self._w = dolfinx.fem.Function(bilinear_form.arguments()[-1].ufl_function_space())
            self._diagonal = dolfinx.fem.petsc.create_vector(self._w.function_space)
            self._vector = dolfinx.fem.petsc.create_vector(self._w.function_space)
            self._vector_product = dolfinx.fem.form(ufl.action(bilinear_form, self._w))
            self._compiled_diagonal = dolfinx.fem.form(
                bilinear_form,
                form_compiler_options=diagnal_options,
                jit_options=jit_options,
            )

    def mult(self, mat, X, Y):
        """Compute Y = A * X, where A is the bilinear form.

        Note:
            This method never assembles the full matrix A.

        Args:
            mat: The PETSc matrix (not used).
            X: The input vector.
            Y: The output vector.

        """
        # Move data into local working array

        dolfinx.fem.petsc.assign(X, self._w)

        # Zero out any input from Dirichlet BCs
        if isinstance(self._compiled_diagonal, dolfinx.fem.Form):
            bcs0 = self._bcs
            for bc in self._bcs:
                di = bc.dof_indices()
                odi = di[0][: di[1]]
                self._w.x.array[odi] = 0
            self._w.x.scatter_forward()

        else:
            bcs0 = dolfinx.fem.bcs_by_block(
                dolfinx.fem.extract_function_spaces(self._compiled_diagonal), self._bcs
            )
            for i, bcs in enumerate(bcs0):
                for bc in bcs:
                    di = bc.dof_indices()
                    odi = di[0]
                    self._w[i].x.array[odi] = 0
                self._w[i].x.scatter_forward()

        # Assemble action
        with self._vector.localForm() as loc:
            loc.set(0)

        dolfinx.fem.petsc.assemble_vector(self._vector, self._vector_product)
        dolfinx.la.petsc._ghost_update(
            self._vector, PETSc.InsertMode.ADD, PETSc.ScatterMode.REVERSE
        )

        # Insert X at Dirichlet dofs
        if isinstance(self._compiled_diagonal, dolfinx.fem.Form):
            bcs0 = self._bcs
            for bc in self._bcs:
                di = bc.dof_indices()
                odi = di[0][: di[1]]
                self._vector.array_w[odi] = X.array_r[odi]
        else:
            bcs0 = dolfinx.fem.bcs_by_block(
                dolfinx.fem.extract_function_spaces(self._compiled_diagonal), self._bcs
            )
            offset0, _ = self._vector.getAttr("_blocks")
            for bcs, off0, off1 in zip(bcs0, offset0[:-1], offset0[1:], strict=True):  # type: ignore[assignment]
                v_array = self._vector.array_w[off0:off1]
                x_array = X.array_r[off0:off1]
                for bc in bcs:
                    di = bc.dof_indices()
                    odi = di[0][: di[1]]
                    v_array[odi] = x_array[odi]
        dolfinx.la.petsc._ghost_update(
            self._vector, PETSc.InsertMode.INSERT, PETSc.ScatterMode.FORWARD
        )
        Y.setArray(self._vector)
        Y.ghostUpdate(PETSc.InsertMode.INSERT, PETSc.ScatterMode.FORWARD)

    def getDiagonal(self, mat, vec):
        """Compute the diagonal of the bilinear form.

        Args:
            mat: The PETSc matrix (not used).
            vec: The output vector to store the diagonal.
        """
        # NOTE: Only have to go through a DOLFINx vector due to a
        # bug in PETSc, similar to:
        # https://gitlab.com/petsc/petsc/-/issues/1645
        with self._diagonal.localForm() as loc:
            loc.set(0)
        dolfinx.fem.petsc.assemble_vector(self._diagonal, self._compiled_diagonal)
        self._diagonal.ghostUpdate(PETSc.InsertMode.ADD, PETSc.ScatterMode.REVERSE)
        if isinstance(self._compiled_diagonal, dolfinx.fem.Form):
            for bc in self._bcs:
                di = bc.dof_indices()
                odi = di[0][: di[1]]
                self._diagonal.array_w[odi] = 1
        else:
            bcs0 = dolfinx.fem.bcs_by_block(
                dolfinx.fem.extract_function_spaces(self._compiled_diagonal), self._bcs
            )
            offset0, _ = self._diagonal.getAttr("_blocks")
            for bcs, off0 in zip(bcs0, offset0[:-1], strict=True):  # type: ignore[assignment]
                for bc in bcs:
                    di = bc.dof_indices()
                    odi = di[0][: di[1]]
                    self._diagonal.array_w[off0 + odi] = 1
        self._diagonal.ghostUpdate(PETSc.InsertMode.INSERT, PETSc.ScatterMode.FORWARD)
        vec.setArray(self._diagonal)


# ## Setting up a krylov subspace solver with the matrix-free operator
# As we will solve the problem below with different representations of the
# bilinear form,
# we provide a convenience function for attaching the matrix-free operator
# to a PETSc KSP object.


def attach_matrix_free_operator(
    ksp: PETSc.KSP,
    bilinear_form: ufl.Form,
    bcs: list[dolfinx.fem.DirichletBC] | None = None,
):
    """Attach a matrix-free operator to a PETSc KSP object.

    Args:
        ksp: The PETSc KSP object.
        bilinear_form: The bilinear form.
        bcs: A list of Dirichlet boundary conditions.
    """
    # Check if we have something from a mixed function space
    operator = MatrixFreeOperator(bilinear_form, bcs=bcs)

    A = PETSc.Mat().create(ksp.getComm().tompi4py())
    sizes = operator._diagonal.getSizes()
    A.setSizes([sizes, sizes])

    A.setType("python")
    A.setPythonContext(operator)
    A.assemble()
    ksp.setOperators(A)


# We are ready to solve our problem.
# In this demo we will consider two approaches, using
# `basix.ufl.mixed_element`and using a `ufl.MixedFunctionSpace`.

# We start by definng our mesh and finite element spaces.

N = 25
mesh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, N, N)
el_0 = basix.ufl.element("Lagrange", mesh.basix_cell(), 2, shape=(2,), dtype=dolfinx.default_real_type)
el_1 = basix.ufl.element("Lagrange", mesh.basix_cell(), 1,  dtype=dolfinx.default_real_type)

# We define the analytical solutions `f` and `g`

x = ufl.SpatialCoordinate(mesh)
f = ufl.as_vector((ufl.cos(2 * x[0]) * ufl.sin(x[1]), x[1]))
g = ufl.sin(3 * x[0]) * ufl.cos(4 * x[1])


# Next, we create a general function to extract the bilinear and
# linear forms from the weak formulation.


def extract_system(
    W: dolfinx.fem.FunctionSpace | ufl.MixedFunctionSpace,
    f: ufl.core.expr.Expr,
    g: ufl.core.expr.Expr,
) -> tuple[ufl.Form, ufl.Form]:
    """Extract the bilinear and linear forms."""
    u_h, p_h = ufl.TrialFunctions(W)
    v, q = ufl.TestFunctions(W)
    residual = ufl.inner(u_h - f, v) * ufl.dx + ufl.inner(p_h - g, q) * ufl.dx
    return ufl.system(residual)


# We also define a convenience function for creating the Krylov subspace
# solver and attaching the matrix free operator.


def define_matrix_free_ksp(
    a: ufl.Form,
    bcs: list[dolfinx.fem.DirichletBC],
    prefix: str,
) -> PETSc.KSP:
    comm = a.ufl_domain().ufl_cargo().comm
    ksp = PETSc.KSP().create(comm)
    attach_matrix_free_operator(ksp, a, bcs=bcs)

    ksp.setMonitor(
        lambda _, its, rnorm: PETSc.Sys.Print(f"{prefix} Iter: {its}, rel. residual: {rnorm:.5e}")
    )
    ksp.setType("cg")
    pc = ksp.getPC()
    pc.setType("jacobi")
    dtype = dolfinx.default_scalar_type
    tol = 1e-10 if dtype == np.float64 else 1e-6
    ksp.setTolerances(atol=tol, rtol=tol, max_it=300)
    ksp.setErrorIfNotConverged(False)
    return ksp


# ## Mixed-element approach
# We start by defining the function spaces using a mixed element from
# `basix.ufl.mixed_element`.


def mixed_element(
    mesh: dolfinx.mesh.Mesh, f: ufl.core.expr.Expr, g: ufl.core.expr.Expr
) -> tuple[dolfinx.fem.Function, dolfinx.fem.Function]:
    # Define function space for mixed element and extract subspaces
    W = dolfinx.fem.functionspace(mesh, basix.ufl.mixed_element([el_0, el_1]))
    V, _ = W.sub(0).collapse()
    Q, _ = W.sub(1).collapse()

    # Extract bilinear and linear forms
    a, L = extract_system(W, f, g)

    # Set up Dirichlet boundary conditions using the analytical solution.
    u_bc_expr = dolfinx.fem.Expression(f, V.element.interpolation_points)
    u_bc = dolfinx.fem.Function(V)
    u_bc.interpolate(u_bc_expr)
    mesh.topology.create_connectivity(mesh.topology.dim - 1, mesh.topology.dim)
    bc_facets = dolfinx.mesh.exterior_facet_indices(mesh.topology)
    bc_dofs_u = dolfinx.fem.locate_dofs_topological((W.sub(0), V), mesh.topology.dim - 1, bc_facets)
    p_bc_expr = dolfinx.fem.Expression(g, Q.element.interpolation_points)
    p_bc = dolfinx.fem.Function(Q)
    p_bc.interpolate(p_bc_expr)
    bc_dofs_p = dolfinx.fem.locate_dofs_topological((W.sub(1), Q), mesh.topology.dim - 1, bc_facets)
    bcs = [
        dolfinx.fem.dirichletbc(u_bc, bc_dofs_u, W.sub(0)),
        dolfinx.fem.dirichletbc(p_bc, bc_dofs_p, W.sub(1)),
    ]

    # Assemble RHS with boundary conditions
    b = dolfinx.fem.petsc.assemble_vector(dolfinx.fem.form(L))
    dolfinx.fem.petsc.apply_lifting(b, [dolfinx.fem.form(a)], [bcs])
    b.ghostUpdate(PETSc.InsertMode.ADD, PETSc.ScatterMode.REVERSE)
    dolfinx.fem.petsc.set_bc(b, bcs)

    # Setup matrix free KSP
    ksp = define_matrix_free_ksp(a, bcs, "MixedElement")

    # Solve the system
    wh = dolfinx.fem.Function(W)
    ksp.solve(b, wh.x.petsc_vec)
    wh.x.scatter_forward()

    # Extract subspace solutions
    uh = wh.sub(0).collapse()
    ph = wh.sub(1).collapse()
    return uh, ph


# ## MixedFunctionSpace approach
# We can also define the function space using `ufl.MixedFunctionSpace`.
# This approach is more efficient if there are many subspaces, where
# there is little cross coupling.
# It is also more flexible, as each sub-space can be defined on
# different meshes, such as submeshes of codimension 0 and 1.


def mixed_function_space(
    mesh: dolfinx.mesh.Mesh, f: ufl.core.expr.Expr, g: ufl.core.expr.Expr
) -> tuple[dolfinx.fem.Function, dolfinx.fem.Function]:
    # Create mixed function space from two function spaces
    V = dolfinx.fem.functionspace(mesh, el_0)
    Q = dolfinx.fem.functionspace(mesh, el_1)
    W = ufl.MixedFunctionSpace(V, Q)

    # Extract bilinear and linear forms
    a, L = extract_system(W, f, g)

    # Create Dirichlet boundary conditions using the analytical solution.
    u_bc_expr = dolfinx.fem.Expression(f, V.element.interpolation_points)
    u_bc = dolfinx.fem.Function(V)
    u_bc.interpolate(u_bc_expr)
    mesh.topology.create_connectivity(mesh.topology.dim - 1, mesh.topology.dim)
    bc_facets = dolfinx.mesh.exterior_facet_indices(mesh.topology)
    bc_dofs_u = dolfinx.fem.locate_dofs_topological(V, mesh.topology.dim - 1, bc_facets)
    p_bc_expr = dolfinx.fem.Expression(g, Q.element.interpolation_points)
    p_bc = dolfinx.fem.Function(Q)
    p_bc.interpolate(p_bc_expr)
    bc_dofs_p = dolfinx.fem.locate_dofs_topological(Q, mesh.topology.dim - 1, bc_facets)
    bcs = [
        dolfinx.fem.dirichletbc(u_bc, bc_dofs_u),
        dolfinx.fem.dirichletbc(p_bc, bc_dofs_p),
    ]

    # Compile forms and assemble the RHS with boundary conditions
    # The lifting operation never assembles the full matrix A, it instead
    # assemble the local product A_local g_local, where g_local is the
    # local representation of the Dirichlet data.

    L_compiled = dolfinx.fem.form(ufl.extract_blocks(L))
    a_compiled = dolfinx.fem.form(ufl.extract_blocks(a))
    b = dolfinx.fem.petsc.assemble_vector(L_compiled)
    bcs0 = dolfinx.fem.bcs_by_block(dolfinx.fem.extract_function_spaces(L_compiled), bcs)
    dolfinx.fem.petsc.apply_lifting(b, a_compiled, bcs0)
    b.ghostUpdate(PETSc.InsertMode.ADD, PETSc.ScatterMode.REVERSE)
    dolfinx.fem.petsc.set_bc(b, bcs0)
    b.ghostUpdate(PETSc.InsertMode.INSERT, PETSc.ScatterMode.FORWARD)

    # We define the matrix free KSP and solve the linear system
    ksp = define_matrix_free_ksp(a, bcs, "MixedFunctionSpace")
    wh = b.duplicate()
    ksp.solve(b, wh)
    wh.ghostUpdate(PETSc.InsertMode.INSERT, PETSc.ScatterMode.FORWARD)

    #  Assign solution to dolfinx functions
    uh = dolfinx.fem.Function(V)
    ph = dolfinx.fem.Function(Q)
    dolfinx.fem.petsc.assign(wh, [uh, ph])
    return uh, ph


# ## Checking solution accuracy
# We can now solve the problem using both approaches and compare
# the solution accuracy.
# We compute the L2-error between the numerical solution and the
# analytical solution.

# +
u_me, p_me = mixed_element(mesh, f, g)

u_mfs, p_mfs = mixed_function_space(mesh, f, g)


def compute_L2_error(uh, u_ex) -> float:
    """Compute the L2-error between an approximate solution and
    an exact solution.

    Args:
        uh: The approximate solution.
        u_ex: The exact solution.
    """
    error = ufl.inner(uh - u_ex, uh - u_ex) * ufl.dx
    error = dolfinx.fem.assemble_scalar(dolfinx.fem.form(error))
    return np.sqrt(mesh.comm.allreduce(error, op=MPI.SUM))


error_u_me = compute_L2_error(u_me, f)
error_p_me = compute_L2_error(p_me, g)

PETSc.Sys.Print("Mixed element:")
PETSc.Sys.Print(f"L2 error (u): {error_u_me:.5e}")
PETSc.Sys.Print(f"L2 error (p): {error_p_me:.5e}")

error_u_mfs = compute_L2_error(u_mfs, f)
error_p_mfs = compute_L2_error(p_mfs, g)

PETSc.Sys.Print("Mixed function space:")
PETSc.Sys.Print(f"L2 error (u): {error_u_mfs:.5e}")
PETSc.Sys.Print(f"L2 error (p): {error_p_mfs:.5e}")
# -
