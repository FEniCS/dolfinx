
from mpi4py import MPI
from dolfinx.la import matrix_csr, BlockMode
from dolfinx.mesh import create_unit_square
from dolfinx.fem import form, FunctionSpace, assemble_matrix, VectorFunctionSpace, create_sparsity_pattern
from ufl import dx, inner, grad, TestFunction, TrialFunction
import pyamg
import numpy as np
import scipy
np.set_printoptions(linewidth=200)

mesh = create_unit_square(MPI.COMM_WORLD, 200, 200)
Q = VectorFunctionSpace(mesh, ("CG", 1))
u, v = TestFunction(Q), TrialFunction(Q)
a = form(inner(grad(u), grad(v))*dx + inner(u, v)*dx)
sp = create_sparsity_pattern(a)
sp.finalize()
A = matrix_csr(sp, BlockMode.expanded)
assemble_matrix(A, a)

n = len(A.indptr)//2 + 1
cutoff = A.indptr[n]
A = scipy.sparse.csr_matrix((A.data[:cutoff], A.indices[:cutoff], A.indptr[:n]))

ml = pyamg.ruge_stuben_solver(A)                    # construct the multigrid hierarchy
print(ml)                                           # print hierarchy information
b = np.random.rand(A.shape[0])                      # pick a random right hand side
x = ml.solve(b, tol=1e-10)                          # solve Ax=b to a tolerance of 1e-10
print("residual: ", np.linalg.norm(b-A*x))          # compute norm of residual vector
