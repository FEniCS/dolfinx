/* -*- C -*- */
// Copyright (C) 2009 Johan Hake
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
// First added:  2009-09-22
// Last changed: 2011-05-22

//=============================================================================
// SWIG directives for the DOLFIN real kernel module (pre)
//
// The directives in this file are applied _before_ the header files of the
// modules has been loaded.
//=============================================================================

//-----------------------------------------------------------------------------
// Forward declare Parameters
//-----------------------------------------------------------------------------
namespace dolfin
{
  class Parameters;
}

//-----------------------------------------------------------------------------
// Ignore all functions using real*
// We need to hand an explicit typemap for this type
//-----------------------------------------------------------------------------
/*
%ignore real_set(uint n, real* x, const real* y);
%ignore real_set(uint n, real* x, const real& value);
%ignore real_zero(uint n, real* x);
%ignore real_add(uint n, real* x, const real* y);
%ignore real_sub(uint n, real* x, const real* y);
%ignore real_axpy(uint n, real* x, const real& a, const real* y);
%ignore real_mult(uint n, real* x, const real& a);
%ignore real_div(uint n, real* x, const real& a);
%ignore real_inner(uint n, const real* x, const real* y);
%ignore real_max_abs(uint n, const real* x);
%ignore real_norm(uint n, const real* x);
%ignore real_identity(uint n, real* A, real value=1.0);
*/

//-----------------------------------------------------------------------------
// Global modifications to the Array interface
//-----------------------------------------------------------------------------
%ignore dolfin::Array::operator=;
%ignore dolfin::Array::operator[];

//-----------------------------------------------------------------------------
// Global modifications to the IndexSet interface
//-----------------------------------------------------------------------------
%ignore dolfin::IndexSet::operator[];

//-----------------------------------------------------------------------------
// Global modifications to the dolfin::Set interface
//-----------------------------------------------------------------------------
%ignore dolfin::Set::operator[];

//-----------------------------------------------------------------------------
// Copy Array construction typemaps from NumPy typemaps
//-----------------------------------------------------------------------------
%typemap(in) (dolfin::uint N, const dolfin::uint* x) = (dolfin::uint _array_dim, dolfin::uint* _array);
%typemap(in) (dolfin::uint N, const int* x) = (dolfin::uint _array_dim, int* _array);
%typemap(in) (dolfin::uint N, const double* x) = (dolfin::uint _array_dim, double* _array);

//-----------------------------------------------------------------------------
// Ignores for Hierarchical
//-----------------------------------------------------------------------------
%ignore dolfin::Hierarchical::operator=;

//-----------------------------------------------------------------------------
// Forward declare Hierarchical template class
//-----------------------------------------------------------------------------
namespace dolfin {
  template<class T>
    class Hierarchical;
}

//-----------------------------------------------------------------------------
// Ignore all foo and rename foo_shared_ptr to foo for SWIG >= 2.0
// and ignore foo_shared_ptr for SWIG < 2.0
//-----------------------------------------------------------------------------
%ignore dolfin::Hierarchical::parent;
%rename(parent) dolfin::Hierarchical::parent_shared_ptr;
%ignore dolfin::Hierarchical::child;
%rename(child) dolfin::Hierarchical::child_shared_ptr;
%ignore dolfin::Hierarchical::coarse;
%rename(coarse) dolfin::Hierarchical::coarse_shared_ptr;
%ignore dolfin::Hierarchical::fine;
%rename(fine) dolfin::Hierarchical::fine_shared_ptr;

