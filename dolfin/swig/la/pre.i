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
// Ignore warnings about nested classes.
//-----------------------------------------------------------------------------
%warnfilter(325) dolfin::GenericLinearAlgebraFactory::NotImplementedLinearOperator;

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
// Modify VectorSpaceBasis::operator[]
//-----------------------------------------------------------------------------
%rename(_sub) dolfin::VectorSpaceBasis::operator[];

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
//%rename(__float__) dolfin::Scalar::operator double;
//%rename(assign) dolfin::Scalar::operator=;

//-----------------------------------------------------------------------------
// Typemaps for GenericMatrix get and set functions
//-----------------------------------------------------------------------------
%typemap(in) const double* block = double* _array;
%typemap(in) (std::size_t m, const dolfin::la_index* rows) = (std::size_t _array_dim, dolfin::la_index* _array);
%typemap(in) (std::size_t n, const dolfin::la_index* cols) = (std::size_t _array_dim, dolfin::la_index* _array);

%typecheck(SWIG_TYPECHECK_DOUBLE_ARRAY) (std::size_t m, const dolfin::la_index* rows)
{
  // rows typemap
  $1 = PyArray_Check($input);
}

%typecheck(SWIG_TYPECHECK_DOUBLE_ARRAY) (std::size_t n, const dolfin::la_index* cols)
{
  // cols typemap
  $1 = PyArray_Check($input);
}

//-----------------------------------------------------------------------------
// Ignore low level interface
//-----------------------------------------------------------------------------
%ignore dolfin::LinearAlgebraObject::instance;
%ignore dolfin::GenericTensor::get(double*, const  dolfin::la_index*, const dolfin::la_index * const *) const;
%ignore dolfin::GenericTensor::set(const double* , const dolfin::la_index* , const dolfin::la_index * const *);
%ignore dolfin::GenericTensor::add(const double* , const dolfin::la_index* , const dolfin::la_index * const *);
%ignore dolfin::PETScLinearOperator::wrapper;

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
%ignore dolfin::GenericVector::get(double*, std::size_t, const dolfin::la_index*) const;
%ignore dolfin::GenericVector::set(const double* , std::size_t m, const dolfin::la_index*);

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

%ignore dolfin::GenericMatrix::set(const double*, const dolfin::la_index*,
                                   const dolfin::la_index * const *);
%ignore dolfin::GenericMatrix::add(const double*, const dolfin::la_index*,
                                   const dolfin::la_index * const * rows);
%ignore dolfin::GenericMatrix::get(double*, const dolfin::la_index*,
                                   const dolfin::la_index * const *) const;
%ignore dolfin::GenericMatrix::data;
%ignore dolfin::GenericMatrix::getitem;
%ignore dolfin::GenericMatrix::setitem;
%ignore dolfin::GenericMatrix::operator();

//-----------------------------------------------------------------------------
// PETSc/SLEPc backend
//-----------------------------------------------------------------------------
#ifdef HAS_PETSC
// Ignore MatNullSpace not properly wrapped by SWIG
%ignore dolfin::PETScPreconditioner::near_nullspace() const;

// Only ignore C++ accessors if petsc4py is enabled
#ifdef HAS_PETSC4PY
%ignore dolfin::PETScVector::vec() const;
%ignore dolfin::PETScBaseMatrix::mat() const;
%ignore dolfin::PETScKrylovSolver::ksp() const;
%ignore dolfin::PETScLUSolver::ksp() const;
%ignore dolfin::PETScSNESSolver::snes() const;
%ignore dolfin::PETScTAOSolver::tao() const;
#else
// Ignore everything
%ignore dolfin::PETScVector::vec;
%ignore dolfin::PETScBaseMatrix::mat;
%ignore dolfin::PETScKrylovSolver::ksp;
%ignore dolfin::PETScLUSolver::ksp;
%ignore dolfin::PETScSNESSolver::snes;
%ignore dolfin::PETScTAOSolver::tao;
#endif
#endif

#ifdef HAS_SLEPC
// Only ignore C++ accessors if slepc4py is enabled
#ifdef HAS_SLEPC4PY
%ignore dolfin::SLEPcEigenSolver::eps() const;
#else
// Ignore everything
%ignore dolfin::SLEPcEigenSolver::eps;
#endif
#endif

//-----------------------------------------------------------------------------
// Add director classes
//-----------------------------------------------------------------------------
%feature("director") dolfin::PETScUserPreconditioner;

%feature("director") dolfin::PETScLinearOperator;
%feature("nodirector") dolfin::PETScLinearOperator::str;
%feature("nodirector") dolfin::PETScLinearOperator::wrapper;

%feature("director") dolfin::LinearOperator;
%feature("nodirector") dolfin::LinearOperator::instance;
%feature("nodirector") dolfin::LinearOperator::shared_instance;

//-----------------------------------------------------------------------------
// Director typemaps for dolfin::GenericVector
//-----------------------------------------------------------------------------
%typemap(directorin, fragment="NoDelete") dolfin::GenericVector&
{
  // Director in dolfin::GenericVector&
  std::shared_ptr< dolfin::GenericVector > *smartresult = new std::shared_ptr< dolfin::GenericVector >(reference_to_no_delete_pointer($1_name));
  $input = SWIG_NewPointerObj(%as_voidptr(smartresult), $descriptor(std::shared_ptr< dolfin::GenericVector > *), SWIG_POINTER_OWN);
}

%typemap(directorin, fragment="NoDelete") const dolfin::GenericVector&
{
  // Director in const dolfin::GenericVector&
  std::shared_ptr< const dolfin::GenericVector > *smartresult = new std::shared_ptr< const dolfin::GenericVector >(reference_to_no_delete_pointer($1_name));
  $input = SWIG_NewPointerObj(%as_voidptr(smartresult), $descriptor(std::shared_ptr< dolfin::GenericVector > *), SWIG_POINTER_OWN);
}

//-----------------------------------------------------------------------------
// Director typemaps for dolfin::PETScVector
//-----------------------------------------------------------------------------
%typemap(directorin, fragment="NoDelete") dolfin::PETScVector&
{
  // Director in dolfin::PETScVector&
  std::shared_ptr< dolfin::PETScVector > *smartresult
    = new std::shared_ptr< dolfin::PETScVector >(reference_to_no_delete_pointer($1_name));
  $input = SWIG_NewPointerObj(%as_voidptr(smartresult), $descriptor(std::shared_ptr< dolfin::PETScVector > *), SWIG_POINTER_OWN);
}

%typemap(directorin, fragment="NoDelete") const dolfin::PETScVector&
{
  // Director in const dolfin::PETScVector&
  std::shared_ptr< const dolfin::PETScVector > *smartresult
    = new std::shared_ptr< const dolfin::PETScVector >(reference_to_no_delete_pointer($1_name));
  $input = SWIG_NewPointerObj(%as_voidptr(smartresult), $descriptor(std::shared_ptr< dolfin::PETScVector > *), SWIG_POINTER_OWN);
}

//-----------------------------------------------------------------------------
// Director typemaps for dolfin::TpetraVector - just copied from above
// for PETSc. No idea if this is right.
// -----------------------------------------------------------------------------
%typemap(directorin, fragment="NoDelete") dolfin::TpetraVector&
{
  // Director in dolfin::TpetraVector&
  std::shared_ptr< dolfin::TpetraVector > *smartresult
    = new std::shared_ptr< dolfin::TpetraVector >(reference_to_no_delete_pointer($1_name));
  $input = SWIG_NewPointerObj(%as_voidptr(smartresult), $descriptor(std::shared_ptr< dolfin::TpetraVector > *), SWIG_POINTER_OWN);
}

%typemap(directorin, fragment="NoDelete") const dolfin::TpetraVector&
{
  // Director in const dolfin::TpetraVector&
  std::shared_ptr< const dolfin::TpetraVector > *smartresult
    = new std::shared_ptr< const dolfin::TpetraVector >(reference_to_no_delete_pointer($1_name));
  $input = SWIG_NewPointerObj(%as_voidptr(smartresult), $descriptor(std::shared_ptr< dolfin::TpetraVector > *), SWIG_POINTER_OWN);
}
