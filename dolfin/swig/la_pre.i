/* -*- C -*- */
// Copyright (C) 2009 Johan Jansson
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Anders Logg, 2007-2009.
// Modified by Garth Wells, 2008-2009
// Modified by Ola Skavhaug, 2008-2009
// Modified by Kent-Andre Mardal, 2008. 
// Modified by Martin Sandve Alnaes, 2008. 
// Modified by Johan Hake, 2008-2009.
//
// First added:  2007-01-21
// Last changed: 2009-10-07

//=============================================================================
// SWIG directives for the DOLFIN la kernel module (pre)
//
// The directives in this file are applied _before_ the header files of the
// modules has been loaded.
//=============================================================================

//-----------------------------------------------------------------------------
// Fix problem with missing uBLAS namespace
//-----------------------------------------------------------------------------
%inline %{
  namespace boost{ namespace numeric{ namespace ublas{}}}
%}

//-----------------------------------------------------------------------------
// Ignore some operator=
//-----------------------------------------------------------------------------
%ignore dolfin::GenericTensor::operator=;
%ignore dolfin::BlockVector::operator=;
%ignore dolfin::SubVector::operator=;
%ignore dolfin::SubMatrix::operator=;

//-----------------------------------------------------------------------------
// Modify the Scalar interface
//-----------------------------------------------------------------------------
%rename(__float__) dolfin::Scalar::operator double;
%rename(assign) dolfin::Scalar::operator=;


//-----------------------------------------------------------------------------
// Typemaps for GenericMatrix get and set functions
//-----------------------------------------------------------------------------
%typemap(in) const double* block = double* _array;
%typemap(in) (dolfin::uint m, const dolfin::uint* rows) = (dolfin::uint _array_dim, dolfin::uint* _array);
%typemap(in) (dolfin::uint n, const dolfin::uint* cols) = (dolfin::uint _array_dim, dolfin::uint* _array);

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

//-----------------------------------------------------------------------------
// Ignore low level interface from GenericTensor class
//-----------------------------------------------------------------------------
%ignore dolfin::GenericTensor::get(double*, const uint*, const uint * const *) const;
%ignore dolfin::GenericTensor::set(const double* , const uint* , const uint * const *);
%ignore dolfin::GenericTensor::add(const double* , const uint* , const uint * const *);
%ignore dolfin::GenericTensor::instance;

//-----------------------------------------------------------------------------
%ignore dolfin::uBLASVector::operator ()(uint i) const;
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
// Ignore wrapping of the Set variable (Might add typemap for this in future...)
//-----------------------------------------------------------------------------
%ignore dolfin::SparsityPattern::pattern;

//-----------------------------------------------------------------------------
// Declare newobject for vector and matrix get functions
//-----------------------------------------------------------------------------
%newobject _get_vector_sub_vector;
%newobject _get_matrix_sub_vector;
%newobject _get_matrix_sub_matrix;

//-----------------------------------------------------------------------------
// Define a macros for the linear algebra factory interface
//-----------------------------------------------------------------------------
%define LA_PRE_FACTORY(FACTORY_TYPE)
%newobject dolfin::FACTORY_TYPE::create_matrix;
%newobject dolfin::FACTORY_TYPE::create_pattern;
%newobject dolfin::FACTORY_TYPE::create_vector;
%enddef

//-----------------------------------------------------------------------------
// Define a macro for the vector interface
//-----------------------------------------------------------------------------
%define LA_PRE_VEC_INTERFACE(VEC_TYPE)
%rename(assign) dolfin::VEC_TYPE::operator=;

%ignore dolfin::VEC_TYPE::operator[];
%ignore dolfin::VEC_TYPE::operator*=;
%ignore dolfin::VEC_TYPE::operator/=;
%ignore dolfin::VEC_TYPE::operator+=;
%ignore dolfin::VEC_TYPE::operator-=;
%ignore dolfin::VEC_TYPE::getitem;
%ignore dolfin::VEC_TYPE::setitem;

//-----------------------------------------------------------------------------
// Ignore the get and set functions used for blocks
// NOTE: The %ignore have to be set using the actuall type used in the declaration
// so we cannot use dolfin::uint or unsigned int for uint. Strange...
//-----------------------------------------------------------------------------
%ignore dolfin::VEC_TYPE::get(double*, uint, const uint*) const;
%ignore dolfin::VEC_TYPE::set(const double* , uint m, const uint*);

%newobject dolfin::VEC_TYPE::copy;

%ignore dolfin::VEC_TYPE::data() const;
%ignore dolfin::VEC_TYPE::data();

%enddef

//-----------------------------------------------------------------------------
// Define a macro for the matrix interface
//-----------------------------------------------------------------------------
%define LA_PRE_MAT_INTERFACE(MAT_TYPE)
%rename(assign) dolfin::MAT_TYPE::operator=;

%ignore dolfin::MAT_TYPE::operator*=;
%ignore dolfin::MAT_TYPE::operator/=;
%ignore dolfin::MAT_TYPE::operator+=;
%ignore dolfin::MAT_TYPE::operator-=;

%newobject dolfin::MAT_TYPE::copy;

%ignore dolfin::MAT_TYPE::data;
%ignore dolfin::MAT_TYPE::getitem;
%ignore dolfin::MAT_TYPE::setitem;
%ignore dolfin::MAT_TYPE::operator();
%enddef

//-----------------------------------------------------------------------------
// Run the macros for default uBLAS backend
//-----------------------------------------------------------------------------
LA_PRE_VEC_INTERFACE(GenericVector)
LA_PRE_VEC_INTERFACE(Vector)
LA_PRE_VEC_INTERFACE(uBLASVector)

LA_PRE_MAT_INTERFACE(GenericMatrix)
LA_PRE_MAT_INTERFACE(Matrix)
LA_PRE_MAT_INTERFACE(uBLASMatrix<dolfin::ublas_sparse_matrix>)
LA_PRE_MAT_INTERFACE(uBLASMatrix<dolfin::ublas_dense_matrix>)

LA_PRE_FACTORY(DefaultFactory)
LA_PRE_FACTORY(uBLASFactory<dolfin::ublas_sparse_matrix>)
LA_PRE_FACTORY(uBLASFactory<dolfin::ublas_dense_matrix>)

//-----------------------------------------------------------------------------
// Run macros for PETSc backend
//-----------------------------------------------------------------------------
#ifdef HAS_PETSC
LA_PRE_VEC_INTERFACE(PETScVector)
LA_PRE_MAT_INTERFACE(PETScMatrix)
LA_PRE_FACTORY(PETScFactory)
#endif

//-----------------------------------------------------------------------------
// Run macros for Trilinos backend
//-----------------------------------------------------------------------------
#ifdef HAS_TRILINOS
LA_PRE_VEC_INTERFACE(EpetraVector)
LA_PRE_MAT_INTERFACE(EpetraMatrix)
LA_PRE_FACTORY(EpetraFactory)

//-----------------------------------------------------------------------------
// Create in and out typemaps for boost::shared_ptr<Foo>
//-----------------------------------------------------------------------------
FOREIGN_SHARED_PTR_TYPEMAPS(Epetra_FECrsMatrix)
FOREIGN_SHARED_PTR_TYPEMAPS(Epetra_FEVector)
#endif

//-----------------------------------------------------------------------------
// Run macros for MTL4 backend
//-----------------------------------------------------------------------------
#ifdef HAS_MTL4
LA_PRE_VEC_INTERFACE(MTL4Vector)
LA_PRE_MAT_INTERFACE(MTL4Matrix)
LA_PRE_FACTORY(MTL4Factory)
%ignore dolfin::MTL4Vector::vec;
%ignore dolfin::MTL4Matrix::mat;
#endif

//-----------------------------------------------------------------------------
// Add director classes
//-----------------------------------------------------------------------------
%feature("director") dolfin::PETScKrylovMatrix;
%feature("director") dolfin::uBLASKrylovMatrix;





