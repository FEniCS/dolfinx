/* -*- C -*- */
// Copyright (C) 2007-2009 Anders Logg
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Ola Skavhaug, 2008
// Modified by Martin Sandve Alnaes, 2008
// Modified by Johan Hake, 2008-2009
// Modified by Garth Wells, 2008-2009
// Modified by Kent-Andre Mardal, 2009
// 
// First added:  2007-08-16
// Last changed: 2009-11-29

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
%rename (_function_space) dolfin::Function::function_space_ptr;
%rename(_sub) dolfin::Function::operator[];
%rename(assign) dolfin::Function::operator=;
%rename(_in) dolfin::Function::in;

//-----------------------------------------------------------------------------
// Modifying the interface of FunctionSpace
//-----------------------------------------------------------------------------
%rename(sub) dolfin::FunctionSpace::operator[];
%rename(assign) dolfin::FunctionSpace::operator=;

//-----------------------------------------------------------------------------
// Rename eval(val, data) method
// We need to rename the method in the base class as the Python callback ends
// up here.
//-----------------------------------------------------------------------------
%rename(eval_data) dolfin::GenericFunction::eval(std::vector<double>& values, const Data& data) const;

//-----------------------------------------------------------------------------
// Ignore the Data.x, pointer to the coordinates in the Data object
//-----------------------------------------------------------------------------
%ignore dolfin::Data::x;
%rename (x) dolfin::Data::x_();

//-----------------------------------------------------------------------------
// Modifying the interface of Constant
//-----------------------------------------------------------------------------
%rename (__float__) dolfin::Constant::operator double() const;
%ignore dolfin::Constant::operator=;

//-----------------------------------------------------------------------------
// Add director classes
//-----------------------------------------------------------------------------
%feature("director") dolfin::Expression;
%feature("nodirector") dolfin::Expression::evaluate;
%feature("nodirector") dolfin::Expression::restrict;
%feature("nodirector") dolfin::Expression::gather;
%feature("nodirector") dolfin::Expression::value_dimension;
%feature("nodirector") dolfin::Expression::value_rank;
