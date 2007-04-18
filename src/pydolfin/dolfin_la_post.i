// Map uBlasMatrix template to Python
%template(uBlasSparseMatrix) dolfin::uBlasMatrix<dolfin::ublas_sparse_matrix>;
%template(uBlasDenseMatrix) dolfin::uBlasMatrix<dolfin::ublas_dense_matrix>;

// Select matrix types (for PETSc or uBlas)
#ifdef HAVE_PETSC_H
%pythoncode
%{
  # Explicit typedefs
  Vector = PETScVector
  Matrix = PETScMatrix
  KrylovSolver = PETScKrylovSolver
  LUSolver = PETScLUSolver

  Vector_createScatterer = PETScVector_createScatterer
  Vector_gather = PETScVector_gather
  Vector_scatter = PETScVector_scatter
%}
#else
%pythoncode
%{
  # Explicit typedefs
  Vector = uBlasVector
  Matrix = uBlasSparseMatrix
  KrylovSolver = uBlasKrylovSolver
  LUSolver = uBlasLUSolver
%}
#endif

// Select matrix types (for both PETSc and uBlas)
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
