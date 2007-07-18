// Can't handle overloading on enums Preconditioner and KrylovMethod
%ignore dolfin::uBlasKrylovSolver;

// Fix problem with missing uBlas namespace
%inline %{
  namespace boost{ namespace numeric{ namespace ublas{}}}
%}

// uBlas dummy classes (need to declare since they are now known)
namespace dolfin {
  class ublas_dense_matrix {};
  class ublas_sparse_matrix {};
  class ublas_vector {};
}
