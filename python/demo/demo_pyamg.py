
from mpi4py import MPI
from dolfinx.la import matrix_csr, BlockMode
from dolfinx.mesh import create_unit_cube
from dolfinx.fem import form, FunctionSpace, assemble_matrix, VectorFunctionSpace, create_sparsity_pattern
from ufl import dx, inner, grad, TestFunction, TrialFunction

from scipy.sparse.linalg import cg

try:
    import pyamg
except ImportError:
    raise ImportError("This demo needs pyamg - try 'pip install pyamg'")

import numpy as np
import scipy
np.set_printoptions(linewidth=200)

mesh = create_unit_cube(MPI.COMM_WORLD, 12, 12, 12)
Q = VectorFunctionSpace(mesh, ("CG", 2))
u, v = TestFunction(Q), TrialFunction(Q)
k = 1.0
a = form(inner(grad(u), grad(v))*dx + k*inner(u, v)*dx)
sp = create_sparsity_pattern(a)
sp.finalize()
A = matrix_csr(sp, BlockMode.expanded)
assemble_matrix(A, a)

A = scipy.sparse.csr_matrix((A.data, A.indices, A.indptr))

# help(pyamg.smoothed_aggregation_solver)

ml = pyamg.smoothed_aggregation_solver(A, strength=('symmetric',{'theta' : 0.08}))
print(ml)                                           # print hierarchy information
b = np.random.rand(A.shape[0])                      # pick a random right hand side
M = ml.aspreconditioner(cycle='V')             # preconditioner

it = 0
def f(x):
    global it
    print(it)
    it += 1

x,info = cg(A, b, tol=1e-12, maxiter=300, M=M, callback=f)   # solve with CG
print("Residual", np.linalg.norm(b-A*x))
