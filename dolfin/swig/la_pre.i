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
// Last changed: 2009-12-16

//=============================================================================
// SWIG directives for the DOLFIN la kernel module (pre)
//
// The directives in this file are applied _before_ the header files of the
// modules has been loaded.
//=============================================================================

//-----------------------------------------------------------------------------
// Ignore the get_eigen{value,pair} methods (reimplemented in post)
//-----------------------------------------------------------------------------
#ifdef HAS_SLEPC
%ignore dolfin::SLEPcEigenSolver::get_eigenvalue;
%ignore dolfin::SLEPcEigenSolver::get_eigenpair;
#endif

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
%ignore dolfin::SubMatrix::operator=;

//-----------------------------------------------------------------------------
// Modify the Scalar interface
//-----------------------------------------------------------------------------
%rename(__float__) dolfin::Scalar::operator double;
%rename(assign) dolfin::Scalar::operator=;

//-----------------------------------------------------------------------------
// Modify the LAPACK interface
//-----------------------------------------------------------------------------
%ignore dolfin::LAPACKVector::operator[];
%ignore dolfin::LAPACKMatrix::operator() (uint i, uint j) const;

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
// Modify the GenericVector interface
//-----------------------------------------------------------------------------
%rename(_assign) dolfin::GenericVector::operator=;

%ignore dolfin::GenericVector::operator[];
%ignore dolfin::GenericVector::operator*=;
%ignore dolfin::GenericVector::operator/=;
%ignore dolfin::GenericVector::operator+=;
%ignore dolfin::GenericVector::operator-=;
%ignore dolfin::GenericVector::getitem;
%ignore dolfin::GenericVector::setitem;

//-----------------------------------------------------------------------------
// Ignore the get and set functions used for blocks
// NOTE: The %ignore has to be set using the actual type used in the declaration
// so we cannot use dolfin::uint or unsigned int for uint. Strange...
//-----------------------------------------------------------------------------
%ignore dolfin::GenericVector::get(double*, uint, const uint*) const;
%ignore dolfin::GenericVector::set(const double* , uint m, const uint*);

%newobject dolfin::GenericVector::copy;

%ignore dolfin::GenericVector::data() const;
%ignore dolfin::GenericVector::data();

//-----------------------------------------------------------------------------
// Modify the GenericMatrix interface
//-----------------------------------------------------------------------------
%rename(assign) dolfin::GenericMatrix::operator=;

%ignore dolfin::GenericMatrix::operator*=;
%ignore dolfin::GenericMatrix::operator/=;
%ignore dolfin::GenericMatrix::operator+=;
%ignore dolfin::GenericMatrix::operator-=;

%newobject dolfin::GenericMatrix::copy;

%ignore dolfin::GenericMatrix::data;
%ignore dolfin::GenericMatrix::getitem;
%ignore dolfin::GenericMatrix::setitem;
%ignore dolfin::GenericMatrix::operator();

//-----------------------------------------------------------------------------
// Modify uBLAS matrices, as these are not renamed by the GenericMatrix rename
//-----------------------------------------------------------------------------
%rename(assign) dolfin::uBLASMatrix<dolfin::ublas_sparse_matrix>::operator=;
%rename(assign) dolfin::uBLASMatrix<dolfin::ublas_dense_matrix>::operator=;

LA_PRE_FACTORY(DefaultFactory)
LA_PRE_FACTORY(uBLASFactory<dolfin::ublas_sparse_matrix>)
LA_PRE_FACTORY(uBLASFactory<dolfin::ublas_dense_matrix>)

//-----------------------------------------------------------------------------
// Run macros for PETSc backend
//-----------------------------------------------------------------------------
#ifdef HAS_PETSC
LA_PRE_FACTORY(PETScFactory)
#endif

//-----------------------------------------------------------------------------
// Run macros for Trilinos backend
//-----------------------------------------------------------------------------
#ifdef HAS_TRILINOS
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
LA_PRE_FACTORY(MTL4Factory)
%ignore dolfin::MTL4Vector::vec;
%ignore dolfin::MTL4Matrix::mat;
#endif

//-----------------------------------------------------------------------------
// Add director classes
//-----------------------------------------------------------------------------
%feature("director") dolfin::PETScKrylovMatrix;
%feature("director") dolfin::uBLASKrylovMatrix;
