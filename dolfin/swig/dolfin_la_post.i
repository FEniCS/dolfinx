// Define uBlas matrix types (typedefs are ignored)
%template(uBlasSparseMatrix) dolfin::uBlasMatrix<dolfin::ublas_sparse_matrix>;
%template(uBlasDenseMatrix) dolfin::uBlasMatrix<dolfin::ublas_dense_matrix>;
