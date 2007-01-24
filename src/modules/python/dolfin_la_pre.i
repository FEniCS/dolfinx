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

