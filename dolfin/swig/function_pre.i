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
// Last changed: 2009-10-06

// ===========================================================================
// SWIG directives for the DOLFIN function kernel module (pre)
//
// The directives in this file are applied _before_ the header files of the
// modules has been loaded.
// ===========================================================================

//-----------------------------------------------------------------------------
// Ignore reference (to FunctionSpaces) constructors of Function
//-----------------------------------------------------------------------------
%ignore dolfin::Function::Function(const FunctionSpace&);
%ignore dolfin::Function::Function(const FunctionSpace&, GenericVector&);
%ignore dolfin::Function::Function(const FunctionSpace&, std::string);
%ignore dolfin::Function::function_space;
%rename (function_space) dolfin::Function::function_space_ptr;

//-----------------------------------------------------------------------------
// Modifying the interface of Function
//-----------------------------------------------------------------------------
%rename(_sub) dolfin::Function::operator[];
%rename(assign) dolfin::Function::operator=;
%rename(_in) dolfin::Function::in;

//-----------------------------------------------------------------------------
// Rename eval(val, data) function
//-----------------------------------------------------------------------------
%rename(eval_data) dolfin::GenericFunction::eval(double* values, const Data& data) const;

//-----------------------------------------------------------------------------
// Ignore the Data.x, pointer to the coordinates in the Data object
//-----------------------------------------------------------------------------
%ignore dolfin::Data::x;
%rename (x) dolfin::Data::x_();


//-----------------------------------------------------------------------------
// Turn off value wrapper for std::vector<dolfin::uint>
//-----------------------------------------------------------------------------
%feature("novaluewrapper") std::vector<dolfin::uint>; 

//-----------------------------------------------------------------------------
// Instantiate a dummy std::vector<dolfin::uint> so value wrapper is not used
//-----------------------------------------------------------------------------
%template () std::vector<dolfin::uint>; 

//-----------------------------------------------------------------------------
// Typemap for std::vector<dolfin::uint> value_shape
//-----------------------------------------------------------------------------
%typecheck(SWIG_TYPECHECK_INT32_ARRAY) std::vector<dolfin::uint> value_shape
{
  $1 = PyList_Check($input) ? 1 : 0;
}

%typemap (in) std::vector<dolfin::uint> value_shape
{
  if (PyList_Check($input))
  {  
    PyObject * py_item = 0;
    int size = PyList_Size($input);
    int item = 0;
    $1.reserve(size);
    for (int i = 0; i < size; i++)
    {
      py_item = PyList_GetItem($input,i);
      if (!PyInteger_Check(py_item))
	SWIG_exception(SWIG_TypeError, "expected list of positive int");
      item = static_cast<int>(PyInt_AsLong(py_item));
      if (item < 0)
	SWIG_exception(SWIG_TypeError, "expected list of positive int");
      $1.push_back(item);
    }
  }
  else
  {
    SWIG_exception(SWIG_TypeError, "expected list of positive int");
  }
}
