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
// Last changed: 2012-01-20

//=============================================================================
// SWIG directives for the DOLFIN common kernel module (pre)
//
// The directives in this file are applied _before_ the header files of the
// modules has been loaded.
//=============================================================================

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
%typemap(in) (std::size_t N, const std::size_t* x) = (std::size_t _array_dim, std::size_t* _array);
%typemap(in) (std::size_t N, const int* x) = (std::size_t _array_dim, int* _array);
%typemap(in) (std::size_t N, const double* x) = (std::size_t _array_dim, double* _array);

//-----------------------------------------------------------------------------
// Ignores for Hierarchical
//-----------------------------------------------------------------------------
%ignore dolfin::Hierarchical::operator=;

//-----------------------------------------------------------------------------
// Ignore all foo and rename foo_shared_ptr to _foo for SWIG >= 2.0
// and ignore foo_shared_ptr for SWIG < 2.0
//-----------------------------------------------------------------------------
%ignore dolfin::Hierarchical::parent;
%rename(parent) dolfin::Hierarchical::parent_shared_ptr;
%ignore dolfin::Hierarchical::child;
%rename(child) dolfin::Hierarchical::child_shared_ptr;
%ignore dolfin::Hierarchical::root_node;
%rename(root_node) dolfin::Hierarchical::root_node_shared_ptr;
%ignore dolfin::Hierarchical::leaf_node;
%rename(leaf_node) dolfin::Hierarchical::leaf_node_shared_ptr;
