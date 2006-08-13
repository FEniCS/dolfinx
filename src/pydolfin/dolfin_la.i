%ignore dolfin::uBlasLUSolver::invert;

%rename(copy) dolfin::GenericVector::operator=;

// Declare dummy uBlas classes

%inline %{
  namespace boost{ namespace numeric{ namespace ublas{}}}
%}

namespace dolfin {
  class ublas_dense_matrix {};
  class ublas_sparse_matrix {};
}

%include "dolfin/Matrix.h"
%include "dolfin/Vector.h"
%include "dolfin/GenericMatrix.h"
%include "dolfin/GenericVector.h"
%include "dolfin/DenseMatrix.h"
%include "dolfin/DenseVector.h"
%include "dolfin/SparseMatrix.h"
%include "dolfin/SparseVector.h"
%include "dolfin/LinearSolver.h"
%include "dolfin/KrylovSolver.h"
%include "dolfin/GMRES.h"
%include "dolfin/LU.h"
%include "dolfin/PETScEigenvalueSolver.h"
%include "dolfin/PETScKrylovMatrix.h"
%include "dolfin/PETScKrylovSolver.h"
%include "dolfin/PETScLinearSolver.h"
%include "dolfin/PETScLUSolver.h"
%include "dolfin/PETScManager.h"
%include "dolfin/PETScMatrix.h"
%include "dolfin/PETScPreconditioner.h"
%include "dolfin/PETScVector.h"
%include "dolfin/ublas.h"
%include "dolfin/uBlasDenseMatrix.h"
%include "dolfin/uBlasDummyPreconditioner.h"
%include "dolfin/uBlasKrylovMatrix.h"
%include "dolfin/uBlasKrylovSolver.h"
%include "dolfin/uBlasLinearSolver.h"
%include "dolfin/uBlasLUSolver.h"
%include "dolfin/uBlasMatrix.h"
%include "dolfin/uBlasILUPreconditioner.h"
%include "dolfin/uBlasPreconditioner.h"
%include "dolfin/uBlasSparseMatrix.h"
%include "dolfin/uBlasVector.h"

%template(uBlasSparseMatrix) dolfin::uBlasMatrix<dolfin::ublas_sparse_matrix>;
%template(uBlasDenseMatrix) dolfin::uBlasMatrix<dolfin::ublas_dense_matrix>;

#ifdef HAVE_PETSC_H
%pythoncode
%{
  # Explicit typedefs
  Vector = PETScVector
  Matrix = PETScMatrix
  KrylovSolver = PETScKrylovSolver
%}
#else
%pythoncode
%{
  # Explicit typedefs
  Vector = uBlasVector
  Matrix = uBlasSparseMatrix
  KrylovSolver = uBlasKrylovSolver
%}
#endif

%pythoncode
%{
  # Explicit typedefs
  DenseVector = uBlasVector

  def __getitem__(self, i):
      return self.get(i)
  def __setitem__(self, i, val):
      self.set(i, val)

  GenericVector.__getitem__ = __getitem__
  GenericVector.__setitem__ = __setitem__

  def __getitem__(self, i):
      return self.get(i[0], i[1])
  def __setitem__(self, i, val):
      self.set(i[0], i[1], val)

  GenericMatrix.__getitem__ = __getitem__
  GenericMatrix.__setitem__ = __setitem__
%}
