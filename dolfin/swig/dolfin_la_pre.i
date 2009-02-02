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
  // ************************** pair TYPEMAP *********************************
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

// Ignore low level interface from GenericTensor class
%ignore dolfin::GenericTensor::get(double*, const uint*, const uint * const *) const;
%ignore dolfin::GenericTensor::set(const double* , const uint* , const uint * const *);
%ignore dolfin::GenericTensor::add;
%ignore dolfin::GenericTensor::instance;

// Define a macro for the vector interface
%define LA_PRE_VEC_INTERFACE(VEC_TYPE)
%rename(assign) dolfin::VEC_TYPE::operator=;

%ignore dolfin::VEC_TYPE::operator*=;
%ignore dolfin::VEC_TYPE::operator/=;
%ignore dolfin::VEC_TYPE::operator+=;
%ignore dolfin::VEC_TYPE::operator-=;

// Ignore the get and set functions used for blocks 
// NOTE: The %ignore have to be set using the actuall type used in the declaration
// so we cannot use dolfin::uint or unsigned int for uint. Strange...
%ignore dolfin::VEC_TYPE::get(double*, uint, const uint*) const;
%ignore dolfin::VEC_TYPE::set(const double* , uint m, const uint*);
		
%ignore dolfin::VEC_TYPE::add;

%newobject dolfin::VEC_TYPE::copy;
%enddef

// Define a macro for the matrix interface
%define LA_PRE_MAT_INTERFACE(MAT_TYPE)
%rename(assign) dolfin::MAT_TYPE::operator=;

%ignore dolfin::MAT_TYPE::operator*=;
%ignore dolfin::MAT_TYPE::operator/=;
%ignore dolfin::MAT_TYPE::operator+=;
%ignore dolfin::MAT_TYPE::operator-=;
%ignore dolfin::MAT_TYPE::add;

%newobject dolfin::MAT_TYPE::copy;

%enddef

// Define a macro for the linear algebra factory interface
%define LA_FACTORY(FACTORY_TYPE)
%newobject dolfin::FACTORY_TYPE::create_matrix();
%newobject dolfin::FACTORY_TYPE::create_pattern();
%newobject dolfin::FACTORY_TYPE::create_vector();

%enddef

// Run the macros with different types
LA_PRE_VEC_INTERFACE(GenericVector)
LA_PRE_VEC_INTERFACE(Vector)
LA_PRE_VEC_INTERFACE(uBLASVector)

LA_PRE_MAT_INTERFACE(GenericMatrix)
LA_PRE_MAT_INTERFACE(Matrix)
LA_PRE_MAT_INTERFACE(uBLASMatrix)

LA_FACTORY(LinearAlgebraFactory)
LA_FACTORY(DefaultFactory)

#ifdef HAS_PETSC
LA_PRE_VEC_INTERFACE(PETScVector)
LA_PRE_MAT_INTERFACE(PETScMatrix)
LA_FACTORY(PETScFactory)
#endif

#ifdef HAS_TRILINOS
LA_PRE_VEC_INTERFACE(EpetraVector)
LA_PRE_MAT_INTERFACE(EpetraMatrix)
LA_FACTORY(EpetraFactory)

// Rename functions returning shared_ptr to underlying matrix/vector
%rename (shared_mat) dolfin::EpetraMatrix::mat;
%rename (mat) dolfin::EpetraMatrix::ref_mat;
%rename (shared_vec) dolfin::EpetraVector::vec;
%rename (vec) dolfin::EpetraVector::ref_vec;
#endif

#ifdef HAS_MTL4
LA_PRE_VEC_INTERFACE(MTL4Vector)
LA_PRE_MAT_INTERFACE(MTL4Matrix)
LA_FACTORY(MTL4Factory)
#endif








