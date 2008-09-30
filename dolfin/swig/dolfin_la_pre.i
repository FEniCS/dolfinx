// Can not handle overloading on enums Preconditioner and KrylovMethod
%ignore dolfin::uBLASKrylovSolver;

// Fix problem with missing uBLAS namespace
%inline %{
  namespace boost{ namespace numeric{ namespace ublas{}}}
%}

// uBLAS dummy classes (need to declare since they are now known)
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

// Define a macros for the linear algebra interface
%define LA_PRE_INTERFACE(VEC_TYPE,MAT_TYPE)
%rename(assign) dolfin::VEC_TYPE::operator=;

%ignore dolfin::VEC_TYPE::operator*=;
%ignore dolfin::VEC_TYPE::operator/=;
%ignore dolfin::VEC_TYPE::operator+=;
%ignore dolfin::VEC_TYPE::operator-=;

%ignore dolfin::MAT_TYPE::operator*=;
%ignore dolfin::MAT_TYPE::operator/=;
%ignore dolfin::MAT_TYPE::operator+=;
%ignore dolfin::MAT_TYPE::operator-=;

%newobject dolfin::VEC_TYPE::copy;
%newobject dolfin::MAT_TYPE::copy;
%newobject dolfin::MAT_TYPE::copy;

%enddef

// Define a macros for the linear algebra factory interface
%define LA_FACTORY(FACTORY_TYPE)
%newobject dolfin::FACTORY_TYPE::create_matrix();
%newobject dolfin::FACTORY_TYPE::create_pattern(); 
%newobject dolfin::FACTORY_TYPE::create_vector();

%enddef

// Run the macros with different types
LA_PRE_INTERFACE(GenericVector,GenericMatrix)
LA_PRE_INTERFACE(Vector,Matrix)
LA_PRE_INTERFACE(uBLASVector,uBLASMatrix)

LA_FACTORY(LinearAlgebraFactory)
LA_FACTORY(DefaultFactory)

#ifdef HAS_PETSC
LA_PRE_INTERFACE(PETScVector,PETScMatrix)
LA_FACTORY(PETScFactory)
#endif

#ifdef HAS_TRILINOS
LA_PRE_INTERFACE(EpetraVector,EpetraMatrix)
LA_FACTORY(EpetraFactory)
#endif

#ifdef HAS_MTL4
LA_PRE_INTERFACE(MTL4Vector,MTL4Matrix)
LA_FACTORY(MTL4Factory)
#endif









