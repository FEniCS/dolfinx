/* -*- C -*- */
// Copyright (C) 2009 Johan Jansson
//
// This file is part of DOLFIN.
//
// DOLFIN is free software: you can redistribute it and/or modify
// it under the terms of the GNU Lesser General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// DOLFIN is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with DOLFIN. If not, see <http://www.gnu.org/licenses/>.
//
// Modified by Anders Logg, 2007-2009.
// Modified by Garth Wells, 2008-2011.
// Modified by Ola Skavhaug, 2008-2009
// Modified by Kent-Andre Mardal, 2008.
// Modified by Martin Sandve Alnaes, 2008.
// Modified by Johan Hake, 2008-2009.
//
// First added:  2007-01-21
// Last changed: 2011-04-20

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
// Rename set and get from BlockFoo
//-----------------------------------------------------------------------------
//%rename(_get) dolfin::BlockVector::get;
//%rename(_set) dolfin::BlockVector::set;
//%rename(_get) dolfin::BlockMatrix::get;
//%rename(_set) dolfin::BlockMatrix::set;

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
%ignore dolfin::SparsityPattern::diagonal_pattern;
%ignore dolfin::SparsityPattern::off_diagonal_pattern;

//-----------------------------------------------------------------------------
// Declare newobject for vector and matrix get functions
//-----------------------------------------------------------------------------
%newobject _get_vector_sub_vector;
//%newobject _get_matrix_sub_vector;
//%newobject _get_matrix_sub_matrix;

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

%newobject dolfin::Vector::copy;
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

%newobject dolfin::Matrix::copy;
%newobject dolfin::GenericMatrix::copy;

%ignore dolfin::GenericMatrix::data;
%ignore dolfin::GenericMatrix::getitem;
%ignore dolfin::GenericMatrix::setitem;
%ignore dolfin::GenericMatrix::operator();

//-----------------------------------------------------------------------------
// Modify uBLAS matrices, as these are not renamed by the GenericMatrix rename
//-----------------------------------------------------------------------------
%rename(assign) dolfin::uBLASMatrix<boost::numeric::ublas::matrix<double> >::operator=;
%rename(assign) dolfin::uBLASMatrix<boost::numeric::ublas::compressed_matrix<double, boost::numeric::ublas::row_major> >::operator=;
%newobject dolfin::uBLASMatrix<boost::numeric::ublas::matrix<double> >::copy;
%newobject dolfin::uBLASMatrix<boost::numeric::ublas::compressed_matrix<double, boost::numeric::ublas::row_major> >::copy;

// NOTE: Silly SWIG complains when running the LA_PRE_FACTORY macro
//LA_PRE_FACTORY(uBLASFactory<uBLASFactory<boost::numeric::ublas::compressed_matrix<double, boost::numeric::ublas::row_major> >)
%newobject dolfin::uBLASFactory<boost::numeric::ublas::compressed_matrix<double, boost::numeric::ublas::row_major> >::create_matrix;
%newobject dolfin::uBLASFactory<boost::numeric::ublas::compressed_matrix<double, boost::numeric::ublas::row_major> >::create_pattern;
%newobject dolfin::uBLASFactory<boost::numeric::ublas::compressed_matrix<double, boost::numeric::ublas::row_major> >::create_vector;
LA_PRE_FACTORY(uBLASFactory<boost::numeric::ublas::matrix<double> >)

LA_PRE_FACTORY(DefaultFactory)

//-----------------------------------------------------------------------------
// Run macros for PETSc backend
//-----------------------------------------------------------------------------
#ifdef HAS_PETSC

// Ignore reference version of constructor
%ignore dolfin::PETScKrylovSolver(std::string, PETScPreconditioner&);
%ignore dolfin::PETScKrylovSolver(std::string, PETScUserPreconditioner&);

LA_PRE_FACTORY(PETScFactory)
%newobject dolfin::PETScMatrix::copy;
%newobject dolfin::PETScVector::copy;
#endif

//-----------------------------------------------------------------------------
// Run macros for Trilinos backend
//-----------------------------------------------------------------------------
#ifdef HAS_TRILINOS
LA_PRE_FACTORY(EpetraFactory)
%rename(_mat) dolfin::EpetraMatrix::mat;
%rename(_vec) dolfin::EpetraVector::vec;
%ignore dolfin::EpetraMatrix::mat;
%ignore dolfin::EpetraVector::vec;
%newobject dolfin::EpetraMatrix::copy;
%newobject dolfin::EpetraVector::copy;

//-----------------------------------------------------------------------------
// Typemaps for Teuchos::RCP (Trilinos backend)
//-----------------------------------------------------------------------------

%define %RCP_to_const_ref_typemap(Type)
%typemap(in) const Type& {
  int res = SWIG_ConvertPtr($input, (void**)&$1, $1_descriptor, 0);
  if (!SWIG_IsOK(res)) {
    Teuchos::RCP<Type> *rcp_ptr;
    int newmem = 0;
    res = SWIG_ConvertPtrAndOwn($input, (void**)&rcp_ptr, $descriptor(Teuchos::RCP<Type>*), 0, &newmem);
    if (!SWIG_IsOK(res))
      SWIG_exception_fail(SWIG_ArgError(res), "in method '$symname', argument $argnum of type '$type'");
    if (rcp_ptr) {
      $1 = rcp_ptr->get();
      if (newmem & SWIG_CAST_NEW_MEMORY)
        delete rcp_ptr;
    }
    else
      $1 = NULL;
  }
  if (!$1)
    SWIG_exception_fail(SWIG_ValueError, "invalid null reference in method '$symname', argument $argnum of type '$type'");
}


%typecheck(SWIG_TYPECHECK_POINTER) const Type& {
  void *dummy;
  int res;
  res = SWIG_ConvertPtr($input, &dummy, $1_descriptor, 0);
  if (!SWIG_IsOK(res)) {
    Teuchos::RCP<Type> *rcp_ptr;
    int newmem = 0;
    res = SWIG_ConvertPtrAndOwn($input, (void**)&rcp_ptr, $descriptor(Teuchos::RCP<Type>*), 0, &newmem);
    if (rcp_ptr && (newmem & SWIG_CAST_NEW_MEMORY))
      delete rcp_ptr;
  }
  $1 = SWIG_CheckState(res);
}
%enddef

%RCP_to_const_ref_typemap(Epetra_CrsGraph);
%RCP_to_const_ref_typemap(Epetra_BlockMap);
#endif

//-----------------------------------------------------------------------------
// Run macros for MTL4 backend
//-----------------------------------------------------------------------------
#ifdef HAS_MTL4
LA_PRE_FACTORY(MTL4Factory)
%newobject dolfin::MTL4Vector::copy;
%newobject dolfin::MTL4Matrix::copy;
%ignore dolfin::MTL4Vector::vec;
%ignore dolfin::MTL4Matrix::mat;
#endif

//-----------------------------------------------------------------------------
// Add director classes
//-----------------------------------------------------------------------------
%feature("director") dolfin::PETScUserPreconditioner;
%feature("director") dolfin::PETScKrylovMatrix;
%feature("director") dolfin::uBLASKrylovMatrix;

//-----------------------------------------------------------------------------
// Director typemaps for dolfin::PETScVector
//-----------------------------------------------------------------------------
%typemap(directorin) dolfin::PETScVector& {
  // Director in dolfin::PETScVector&
  SWIG_SHARED_PTR_QNAMESPACE::shared_ptr< dolfin::PETScVector > *smartresult = new SWIG_SHARED_PTR_QNAMESPACE::shared_ptr< dolfin::PETScVector >(reference_to_no_delete_pointer($1_name));
  $input = SWIG_NewPointerObj(%as_voidptr(smartresult), $descriptor(SWIG_SHARED_PTR_QNAMESPACE::shared_ptr< dolfin::PETScVector > *), SWIG_POINTER_OWN);
}

%typemap(directorin) const dolfin::PETScVector& {
  // Director in const dolfin::PETScVector&
  SWIG_SHARED_PTR_QNAMESPACE::shared_ptr< const dolfin::PETScVector > *smartresult = new SWIG_SHARED_PTR_QNAMESPACE::shared_ptr< const dolfin::PETScVector >(reference_to_no_delete_pointer($1_name));
  $input = SWIG_NewPointerObj(%as_voidptr(smartresult), $descriptor(SWIG_SHARED_PTR_QNAMESPACE::shared_ptr< dolfin::PETScVector > *), SWIG_POINTER_OWN);
}
