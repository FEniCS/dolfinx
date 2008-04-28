// Can not handle overloading on enums Preconditioner and KrylovMethod
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

%typecheck(SWIG_TYPECHECK_DOUBLE_ARRAY) std::pair<dolfin::uint, dolfin::uint> ij {
 $1 = PyTuple_Check($input) ? 1 : 0;
}

%typemap(in) std::pair<dolfin::uint,dolfin::uint> ij (std::pair<dolfin::uint, dolfin::uint> ij) {
   ij.first   = PyLong_AsUnsignedLong(PyTuple_GetItem($input,0));
   ij.second  = PyLong_AsUnsignedLong(PyTuple_GetItem($input,1));
   $1 = ij;
}


%typemap(in) const double* block = double* _array;
%typemap(in) (dolfin::uint m, const dolfin::uint* rows) = (int _array_dim, unsigned int* _array);
%typemap(in) (dolfin::uint n, const dolfin::uint* cols) = (int _array_dim, unsigned int* _array);
%typecheck(SWIG_TYPECHECK_DOUBLE_ARRAY) (dolfin::uint m, const dolfin::uint* rows) 
{
    // rows typemap
    $1 = PyArray_Check($input);
}

%typecheck(SWIG_TYPECHECK_DOUBLE_ARRAY) (dolfin::uint n, const dolfin::uint* cols) 
{
    // cols typemap
    $1 = PyArray_Check($input);
}

%rename(assign) dolfin::GenericVector::operator=;

