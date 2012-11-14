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
// Modified by Anders Logg 2007-2012
// Modified by Garth Wells 2008-2011
// Modified by Ola Skavhaug 2008-2009
// Modified by Kent-Andre Mardal 2008
// Modified by Martin Sandve Alnaes 2008
// Modified by Johan Hake 2008-2009
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
// Rename solve so it wont clash with solve from fem
//-----------------------------------------------------------------------------
%rename(la_solve) dolfin::solve;

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
// Ignore free function norm(), reimplemented in Python
//-----------------------------------------------------------------------------
%ignore dolfin::norm;

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
%typemap(in) (std::size_t m, const std::size_t* rows) = (std::size_t _array_dim, std::size_t* _array);
%typemap(in) (std::size_t n, const std::size_t* cols) = (std::size_t _array_dim, std::size_t* _array);

%typecheck(SWIG_TYPECHECK_DOUBLE_ARRAY) (std::size_t m, const std::size_t* rows)
{
  // rows typemap
  $1 = PyArray_Check($input);
}

%typecheck(SWIG_TYPECHECK_DOUBLE_ARRAY) (std::size_t n, const std::size_t* cols)
{
  // cols typemap
  $1 = PyArray_Check($input);
}

//-----------------------------------------------------------------------------
// Ignore low level interface
//-----------------------------------------------------------------------------
%ignore dolfin::LinearAlgebraObject::instance;
%ignore dolfin::GenericTensor::get(double*, const std::size_t*, const std::size_t * const *) const;
%ignore dolfin::GenericTensor::set(const double* , const std::size_t* , const std::size_t * const *);
%ignore dolfin::GenericTensor::add(const double* , const std::size_t* , const std::size_t * const *);
%ignore dolfin::PETScLinearOperator::wrapper;

//-----------------------------------------------------------------------------
%ignore dolfin::uBLASVector::operator ()(std::size_t i) const;
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
%ignore dolfin::GenericVector::get(double*, std::size_t, const std::size_t*) const;
%ignore dolfin::GenericVector::set(const double* , std::size_t m, const std::size_t*);

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

%ignore dolfin::GenericMatrix::set(const double*, const std::size_t*,
				   const std::size_t * const *);
%ignore dolfin::GenericMatrix::add(const double*, const std::size_t*,
				   const std::size_t * const * rows);
%ignore dolfin::GenericMatrix::get(double*, const std::size_t*,
				   const std::size_t * const *) const;
%ignore dolfin::GenericMatrix::data;
%ignore dolfin::GenericMatrix::getitem;
%ignore dolfin::GenericMatrix::setitem;
%ignore dolfin::GenericMatrix::operator();


//-----------------------------------------------------------------------------
// Modify uBLAS matrices, as these are not renamed by the GenericMatrix rename
//-----------------------------------------------------------------------------
%rename(assign) dolfin::uBLASMatrix<boost::numeric::ublas::matrix<double> >::operator=;
%rename(assign) dolfin::uBLASMatrix<boost::numeric::ublas::compressed_matrix<double, boost::numeric::ublas::row_major> >::operator=;

// Ignore reference version of constructor
%ignore dolfin::PETScKrylovSolver(std::string, PETScPreconditioner&);
%ignore dolfin::PETScKrylovSolver(std::string, PETScUserPreconditioner&);

//-----------------------------------------------------------------------------
// PETSc backend
//-----------------------------------------------------------------------------
#ifdef HAS_PETSC
%ignore dolfin::PETScVector::vec;
%ignore dolfin::PETScBaseMatrix::mat;
#endif

//-----------------------------------------------------------------------------
// Trilinos backend
//-----------------------------------------------------------------------------
#ifdef HAS_TRILINOS
%ignore dolfin::EpetraMatrix::mat;
%ignore dolfin::EpetraVector::vec;
%ignore dolfin::EpetraMatrix(boost::shared_ptr<Epetra_FECrsMatrix> A);

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
%RCP_to_const_ref_typemap(Teuchos::ParameterList);
#endif

//-----------------------------------------------------------------------------
// Add director classes
//-----------------------------------------------------------------------------
%feature("director") dolfin::PETScUserPreconditioner;

%feature("director") dolfin::PETScLinearOperator;
%feature("nodirector") dolfin::PETScLinearOperator::str;
%feature("nodirector") dolfin::PETScLinearOperator::wrapper;

%feature("director") dolfin::uBLASLinearOperator;
%feature("nodirector") dolfin::uBLASLinearOperator::str;

%feature("director") dolfin::LinearOperator;
%feature("nodirector") dolfin::LinearOperator::instance;
%feature("nodirector") dolfin::LinearOperator::shared_instance;

//-----------------------------------------------------------------------------
// Director typemaps for dolfin::GenericVector
//-----------------------------------------------------------------------------
%typemap(directorin, fragment="NoDelete") dolfin::GenericVector& {
  // Director in dolfin::GenericVector&
  SWIG_SHARED_PTR_QNAMESPACE::shared_ptr< dolfin::GenericVector > *smartresult = new SWIG_SHARED_PTR_QNAMESPACE::shared_ptr< dolfin::GenericVector >(reference_to_no_delete_pointer($1_name));
  $input = SWIG_NewPointerObj(%as_voidptr(smartresult), $descriptor(SWIG_SHARED_PTR_QNAMESPACE::shared_ptr< dolfin::GenericVector > *), SWIG_POINTER_OWN);
}

%typemap(directorin, fragment="NoDelete") const dolfin::GenericVector& {
  // Director in const dolfin::GenericVector&
  SWIG_SHARED_PTR_QNAMESPACE::shared_ptr< const dolfin::GenericVector > *smartresult = new SWIG_SHARED_PTR_QNAMESPACE::shared_ptr< const dolfin::GenericVector >(reference_to_no_delete_pointer($1_name));
  $input = SWIG_NewPointerObj(%as_voidptr(smartresult), $descriptor(SWIG_SHARED_PTR_QNAMESPACE::shared_ptr< dolfin::GenericVector > *), SWIG_POINTER_OWN);
}

//-----------------------------------------------------------------------------
// Director typemaps for dolfin::PETScVector
//-----------------------------------------------------------------------------
%typemap(directorin, fragment="NoDelete") dolfin::PETScVector& {
  // Director in dolfin::PETScVector&
  SWIG_SHARED_PTR_QNAMESPACE::shared_ptr< dolfin::PETScVector > *smartresult = new SWIG_SHARED_PTR_QNAMESPACE::shared_ptr< dolfin::PETScVector >(reference_to_no_delete_pointer($1_name));
  $input = SWIG_NewPointerObj(%as_voidptr(smartresult), $descriptor(SWIG_SHARED_PTR_QNAMESPACE::shared_ptr< dolfin::PETScVector > *), SWIG_POINTER_OWN);
}

%typemap(directorin, fragment="NoDelete") const dolfin::PETScVector& {
  // Director in const dolfin::PETScVector&
  SWIG_SHARED_PTR_QNAMESPACE::shared_ptr< const dolfin::PETScVector > *smartresult = new SWIG_SHARED_PTR_QNAMESPACE::shared_ptr< const dolfin::PETScVector >(reference_to_no_delete_pointer($1_name));
  $input = SWIG_NewPointerObj(%as_voidptr(smartresult), $descriptor(SWIG_SHARED_PTR_QNAMESPACE::shared_ptr< dolfin::PETScVector > *), SWIG_POINTER_OWN);
}

//-----------------------------------------------------------------------------
// Director typemaps for dolfin::uBLASVector
//-----------------------------------------------------------------------------
%typemap(directorin, fragment="NoDelete") dolfin::uBLASVector& {
  // Director in dolfin::uBLASVector&
  SWIG_SHARED_PTR_QNAMESPACE::shared_ptr< dolfin::uBLASVector > *smartresult = new SWIG_SHARED_PTR_QNAMESPACE::shared_ptr< dolfin::uBLASVector >(reference_to_no_delete_pointer($1_name));
  $input = SWIG_NewPointerObj(%as_voidptr(smartresult), $descriptor(SWIG_SHARED_PTR_QNAMESPACE::shared_ptr< dolfin::uBLASVector > *), SWIG_POINTER_OWN);
}

%typemap(directorin, fragment="NoDelete") const dolfin::uBLASVector& {
  // Director in const dolfin::uBLASVector&
  SWIG_SHARED_PTR_QNAMESPACE::shared_ptr< const dolfin::uBLASVector > *smartresult = new SWIG_SHARED_PTR_QNAMESPACE::shared_ptr< const dolfin::uBLASVector >(reference_to_no_delete_pointer($1_name));
  $input = SWIG_NewPointerObj(%as_voidptr(smartresult), $descriptor(SWIG_SHARED_PTR_QNAMESPACE::shared_ptr< dolfin::uBLASVector > *), SWIG_POINTER_OWN);
}
