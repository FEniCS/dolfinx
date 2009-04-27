/* -*- C -*-  (not really, but good for syntax highlighting) */

// Cannot handle overloading on enums Preconditioner and KrylovMethod
%ignore dolfin::uBLASKrylovSolver::uBLASKrylovSolver(dolfin::PreconditionerType);

// Fix problem with missing uBLAS namespace
%inline %{
  namespace boost{ namespace numeric{ namespace ublas{}}}
%}

// Typemaps for GenericMatrix getitem and setitem functions
%typecheck(SWIG_TYPECHECK_DOUBLE_ARRAY) std::pair<dolfin::uint, dolfin::uint> ij {
 $1 = PyTuple_Check($input) ? 1 : 0;
}

%typemap(in) std::pair<dolfin::uint,dolfin::uint> ij (std::pair<dolfin::uint, dolfin::uint> ij) {
  if (!PyTuple_Check($input)) {
    PyErr_SetString(PyExc_TypeError,"expected a tuple of size 2");
    return NULL;
  }
  if (PyTuple_Size($input) != 2){
    PyErr_SetString(PyExc_TypeError,"expected a tuple with size 2");	
    return NULL;
  }
   ij.first   = PyLong_AsUnsignedLong(PyTuple_GetItem($input,0));
   ij.second  = PyLong_AsUnsignedLong(PyTuple_GetItem($input,1));
   $1 = ij;
}

// Typemaps for GenericMatrix get and set functions
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
%ignore dolfin::GenericTensor::add(const double* , const uint* , const uint * const *);
%ignore dolfin::GenericTensor::instance;

// Declare newobject for vector and matrix get functions
%newobject _get_vector_sub_vector;
%newobject _get_matrix_sub_vector;
%newobject _get_matrix_sub_matrix;

// Define a macros for the linear algebra factory interface
%define LA_PRE_FACTORY(FACTORY_TYPE)
%newobject dolfin::FACTORY_TYPE::create_matrix;
%newobject dolfin::FACTORY_TYPE::create_pattern; 
%newobject dolfin::FACTORY_TYPE::create_vector;

%enddef

// Define a macro for the vector interface
%define LA_PRE_VEC_INTERFACE(VEC_TYPE)
%rename(assign) dolfin::VEC_TYPE::operator=;

%ignore dolfin::VEC_TYPE::operator*=;
%ignore dolfin::VEC_TYPE::operator/=;
%ignore dolfin::VEC_TYPE::operator+=;
%ignore dolfin::VEC_TYPE::operator-=;
%ignore dolfin::VEC_TYPE::getitem;
%ignore dolfin::VEC_TYPE::setitem;

// Ignore the get and set functions used for blocks 
// NOTE: The %ignore have to be set using the actuall type used in the declaration
// so we cannot use dolfin::uint or unsigned int for uint. Strange...
%ignore dolfin::VEC_TYPE::get(double*, uint, const uint*) const;
%ignore dolfin::VEC_TYPE::set(const double* , uint m, const uint*);
		
%newobject dolfin::VEC_TYPE::copy;

%ignore dolfin::VEC_TYPE::data() const;
%ignore dolfin::VEC_TYPE::data();

%enddef

// Define a macro for the matrix interface
%define LA_PRE_MAT_INTERFACE(MAT_TYPE)
%rename(assign) dolfin::MAT_TYPE::operator=;

%ignore dolfin::MAT_TYPE::operator*=;
%ignore dolfin::MAT_TYPE::operator/=;
%ignore dolfin::MAT_TYPE::operator+=;
%ignore dolfin::MAT_TYPE::operator-=;

%newobject dolfin::MAT_TYPE::copy;

%rename (_data) dolfin::MAT_TYPE::data() const;

%ignore dolfin::MAT_TYPE::getitem;
%ignore dolfin::MAT_TYPE::setitem;
%ignore dolfin::MAT_TYPE::operator();
%enddef

// Run the macros with different types
LA_PRE_VEC_INTERFACE(GenericVector)
LA_PRE_VEC_INTERFACE(Vector)
LA_PRE_VEC_INTERFACE(uBLASVector)

LA_PRE_MAT_INTERFACE(GenericMatrix)
LA_PRE_MAT_INTERFACE(Matrix)
LA_PRE_MAT_INTERFACE(uBLASSparseMatrix)
LA_PRE_MAT_INTERFACE(uBLASDenseMatrix)

LA_PRE_FACTORY(DefaultFactory)
LA_PRE_FACTORY(uBLASFactory<dolfin::ublas_sparse_matrix>)
LA_PRE_FACTORY(uBLASFactory<dolfin::ublas_dense_matrix>)

#ifdef HAS_PETSC
LA_PRE_VEC_INTERFACE(PETScVector)
LA_PRE_MAT_INTERFACE(PETScMatrix)
LA_PRE_FACTORY(PETScFactory)
#endif

#ifdef HAS_TRILINOS
LA_PRE_VEC_INTERFACE(EpetraVector)
LA_PRE_MAT_INTERFACE(EpetraMatrix)
LA_PRE_FACTORY(EpetraFactory)

// Rename functions returning shared_ptr to underlying matrix/vector
%rename (shared_mat) dolfin::EpetraMatrix::mat;
%rename (mat) dolfin::EpetraMatrix::ref_mat;
%rename (shared_vec) dolfin::EpetraVector::vec;
%rename (vec) dolfin::EpetraVector::ref_vec;
#endif

#ifdef HAS_MTL4
//%ignore dolfin::MTL4Factory;
LA_PRE_VEC_INTERFACE(MTL4Vector)
LA_PRE_MAT_INTERFACE(MTL4Matrix)
LA_PRE_FACTORY(MTL4Factory)
#endif








