/* -*- C -*- */
// Copyright (C) 2007-2009 Anders Logg
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Ola Skavhaug, 2008
// Modified by Martin Sandve Alnaes, 2008
// Modified by Johan Hake, 2008-2009
// Modified by Garth Wells, 2008-2010
// Modified by Kent-Andre Mardal, 2009
//
// First added:  2007-08-16
// Last changed: 2011-01-21

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

//-----------------------------------------------------------------------------
// Ingore operator() in GenericFunction, implemented separately in
// the Python interface.
//-----------------------------------------------------------------------------
%ignore dolfin::GenericFunction::operator();

//-----------------------------------------------------------------------------
// Rename eval(val, data) method
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
// Typemap for std::vector<dolfin::uint> value_shape
//-----------------------------------------------------------------------------
%typecheck(SWIG_TYPECHECK_INT32_ARRAY) std::vector<dolfin::uint> value_shape
{
  $1 = PyList_Check($input) ? 1 : 0;
}

%typemap (in) std::vector<dolfin::uint> value_shape (std::vector<dolfin::uint> tmp_vec)
{
  if (PyList_Check($input))
  {
    PyObject * py_item = 0;
    int size = PyList_Size($input);
    int item = 0;
    tmp_vec.reserve(size);
    for (int i = 0; i < size; i++)
    {
      py_item = PyList_GetItem($input,i);
      if (!PyInteger_Check(py_item))
        SWIG_exception(SWIG_TypeError, "expected list of positive int");
      item = static_cast<int>(PyInt_AsLong(py_item));
      if (item < 0)
        SWIG_exception(SWIG_TypeError, "expected list of positive int");
      tmp_vec.push_back(item);
    }
  }
  else
  {
    SWIG_exception(SWIG_TypeError, "expected list of positive int");
  }
  $1 = tmp_vec;
}

//-----------------------------------------------------------------------------
// Typemap for std::vector<TYPE> values
//-----------------------------------------------------------------------------
%typecheck(SWIG_TYPECHECK_DOUBLE_ARRAY) std::vector<double> values
{
  $1 = PyList_Check($input) ? 1 : 0;
}

%typemap (in) std::vector<double> values (std::vector<double> tmp_vec)
{
  if (PyList_Check($input))
  {
    PyObject * py_item = 0;
    int size = PyList_Size($input);
    double item = 0;
    tmp_vec.reserve(size);
    for (int i = 0; i < size; i++)
    {
      py_item = PyList_GetItem($input,i);
      if (!PyFloat_Check(py_item))
        SWIG_exception(SWIG_TypeError, "expected list of floats");
      item = static_cast<double>(PyFloat_AsDouble(py_item));
      tmp_vec.push_back(item);
    }
  }
  else
  {
    SWIG_exception(SWIG_TypeError, "expected list of floats");
  }
  $1 = tmp_vec;
}

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
