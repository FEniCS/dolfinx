/* -*- C -*- */
// Copyright (C) 2007-2009 Anders Logg
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
// Modified by Ola Skavhaug, 2008
// Modified by Martin Sandve Alnaes, 2008
// Modified by Johan Hake, 2008-2011
// Modified by Garth Wells, 2008-2010
// Modified by Kent-Andre Mardal, 2009
//
// First added:  2007-08-16
// Last changed: 2011-01-31

// ===========================================================================
// SWIG directives for the DOLFIN function kernel module (pre)
//
// The directives in this file are applied _before_ the header files of the
// modules has been loaded.
// ===========================================================================

//-----------------------------------------------------------------------------
// Forward declare FiniteElement
//-----------------------------------------------------------------------------
namespace dolfin
{
  class FiniteElement;
}

//-----------------------------------------------------------------------------
// Ignore reference (to FunctionSpaces) constructors of Function
//-----------------------------------------------------------------------------
%ignore dolfin::Function::Function(const FunctionSpace&);
%ignore dolfin::Function::Function(const FunctionSpace&, GenericVector&);
%ignore dolfin::Function::Function(const FunctionSpace&, std::string);


//-----------------------------------------------------------------------------
// Modifying the interface of Function
//-----------------------------------------------------------------------------
%ignore dolfin::Function::function_space;
%rename(_function_space) dolfin::Function::function_space_ptr;
%rename(_sub) dolfin::Function::operator[];
%rename(assign) dolfin::Function::operator=;
%rename(_in) dolfin::Function::in;

//-----------------------------------------------------------------------------
// Modifying the interface of FunctionSpace
//-----------------------------------------------------------------------------
%rename(sub) dolfin::FunctionSpace::operator[];
%rename(assign) dolfin::FunctionSpace::operator=;
%ignore dolfin::FunctionSpace::collapse() const;

//-----------------------------------------------------------------------------
// Ingore operator() in GenericFunction, implemented separately in
// the Python interface.
//-----------------------------------------------------------------------------
%ignore dolfin::GenericFunction::operator();

//-----------------------------------------------------------------------------
// Rename eval(val, x, cell) method
// We need to rename the method in the base class as the Python callback ends
// up here.
//-----------------------------------------------------------------------------
%rename(eval_cell) dolfin::GenericFunction::eval(Array<double>& values,
                                                 const Array<double>& x,
                                                 const ufc::cell& cell) const;

//-----------------------------------------------------------------------------
// Modifying the interface of Constant
//-----------------------------------------------------------------------------
%rename (__float__) dolfin::Constant::operator double() const;
%rename(assign) dolfin::Constant::operator=;

//-----------------------------------------------------------------------------
// Add director classes
//-----------------------------------------------------------------------------
%feature("director") dolfin::Expression;
%feature("nodirector") dolfin::Expression::evaluate;
%feature("nodirector") dolfin::Expression::restrict;
%feature("nodirector") dolfin::Expression::gather;
%feature("nodirector") dolfin::Expression::value_dimension;
%feature("nodirector") dolfin::Expression::value_rank;
%feature("nodirector") dolfin::Expression::str;
%feature("nodirector") dolfin::Expression::compute_vertex_values;

//-----------------------------------------------------------------------------
// Instantiate Hierarchical FunctionSpace, Function template class
//-----------------------------------------------------------------------------
namespace dolfin {
  class FunctionSpace;
  class Function;
}

%template (HierarchicalFunctionSpace) dolfin::Hierarchical<dolfin::FunctionSpace>;
%template (HierarchicalFunction) dolfin::Hierarchical<dolfin::Function>;
