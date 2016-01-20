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
// Modified by Garth Wells, 2008-2011
// Modified by Kent-Andre Mardal, 2009
//
// First added:  2007-08-16
// Last changed: 2011-11-22

// ===========================================================================
// SWIG directives for the DOLFIN function kernel module (pre)
//
// The directives in this file are applied _before_ the header files of the
// modules has been loaded.
// ===========================================================================

//-----------------------------------------------------------------------------
// Modifying the interface of Function
//-----------------------------------------------------------------------------
%rename(_function_space) dolfin::GenericFunction::function_space;
%rename(sub) dolfin::Function::operator[];
%rename(_assign) dolfin::Function::operator=;
%rename(_in) dolfin::Function::in;

//-----------------------------------------------------------------------------
// Modifying the interface of FunctionSpace
//-----------------------------------------------------------------------------
%rename(assign) dolfin::FunctionSpace::operator=;
%ignore dolfin::FunctionSpace::operator[];
%ignore dolfin::FunctionSpace::collapse() const;

//-----------------------------------------------------------------------------
// Modifying the interface of Function
//-----------------------------------------------------------------------------
%ignore dolfin::FunctionAXPY::pair;
%rename(__add__) dolfin::FunctionAXPY::operator+;
%rename(__sub__) dolfin::FunctionAXPY::operator-;
%rename(__mul__) dolfin::FunctionAXPY::operator*;
%rename(__div__) dolfin::FunctionAXPY::operator/;

//-----------------------------------------------------------------------------
// Rename [] for SpecialFacetFunction -> _sub
//-----------------------------------------------------------------------------
%rename(_sub) dolfin::SpecialFacetFunction::operator[];

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
%feature("nodirector") dolfin::Expression::update;
%feature("nodirector") dolfin::Expression::value_dimension;
%feature("nodirector") dolfin::Expression::value_rank;
%feature("nodirector") dolfin::Expression::str;
%feature("nodirector") dolfin::Expression::compute_vertex_values;
%feature("nodirector") dolfin::Expression::function_space;

//-----------------------------------------------------------------------------
// Macro for defining an in typemap for a const std::vector<std::pair<double, TYPE*> >&
// The typemaps takes a list of tuples of floats and TYPE
//
// TYPE       : The Pointer type
//-----------------------------------------------------------------------------
%define IN_TYPEMAPS_STD_VECTOR_OF_PAIRS_OF_DOUBLE_AND_POINTER(TYPE)

//-----------------------------------------------------------------------------
// Make SWIG aware of the shared_ptr version of TYPE
//-----------------------------------------------------------------------------
%types(std::shared_ptr<dolfin::TYPE>*);

//-----------------------------------------------------------------------------
// Run the macros for the combination of const and no const of
// {const} std::vector<std::pair<double, {const} dolfin::TYPE *> >
//-----------------------------------------------------------------------------
CONST_IN_TYPEMAPS_STD_VECTOR_OF_PAIRS_OF_DOUBLE_AND_POINTER(TYPE,const)
CONST_IN_TYPEMAPS_STD_VECTOR_OF_PAIRS_OF_DOUBLE_AND_POINTER(TYPE,)

%enddef

//-----------------------------------------------------------------------------
// Macro for defining in typemaps for
// {const} std::vector<std::pair<double, {const} dolfin::TYPE *> >
// using a Python List of TYPE
//-----------------------------------------------------------------------------
%define CONST_IN_TYPEMAPS_STD_VECTOR_OF_PAIRS_OF_DOUBLE_AND_POINTER(TYPE,CONST)

//-----------------------------------------------------------------------------
// The typecheck
//-----------------------------------------------------------------------------
%typecheck(SWIG_TYPECHECK_POINTER) std::vector<std::pair<double, CONST dolfin::TYPE *> >
{
  $1 = PyList_Check($input) ? 1 : 0;
}

//-----------------------------------------------------------------------------
// The {const} std::vector<std::pair<double, {const} dolfin::TYPE *> > typemap
//-----------------------------------------------------------------------------
%typemap (in) std::vector<std::pair<double, CONST dolfin::TYPE *> >
  (std::vector<std::pair<double, CONST dolfin::TYPE *> > tmp_vec,
   std::shared_ptr<dolfin::TYPE> tempshared,
   dolfin::TYPE* arg)
{

  // CONST_IN_TYPEMAPS_STD_VECTOR_OF_PAIRS_OF_DOUBLE_AND_POINTER(TYPE, CONST)
  if (!PyList_Check($input))
    SWIG_exception(SWIG_TypeError, "list of TYPE expected");

  int size = PyList_Size($input);
  int res = 0;
  PyObject * py_item = 0;
  void * itemp = 0;
  int newmem = 0;
  double value;
  tmp_vec.reserve(size);
  for (int i = 0; i < size; i++)
  {
    // Get python object
    py_item = PyList_GetItem($input, i);

    // Check that it is a tuple with size 2 and first item is a float
    if (!PyTuple_Check(py_item) || PyTuple_Size(py_item) != 2 || \
	!PyFloat_Check(PyTuple_GetItem(py_item, 0)))
      SWIG_exception(SWIG_TypeError, "list of tuples of float and TYPE expected.");

    // Get double value
    value = PyFloat_AsDouble(PyTuple_GetItem(py_item, 0));

    // Try convert the second tuple argument
    res = SWIG_ConvertPtr(PyTuple_GetItem(py_item, 1), \
			  &itemp, $descriptor(dolfin::TYPE *), 0);

    if (SWIG_IsOK(res))
    {
      // We have the pointer and the value, push them back!
      tmp_vec.push_back(std::make_pair(value, reinterpret_cast<dolfin::TYPE *>(itemp)));
    }
    else
    {
      // If failed with normal pointer conversion then try with shared_ptr conversion
      newmem = 0;
      res = SWIG_ConvertPtrAndOwn(PyTuple_GetItem(py_item, 1), &itemp, $descriptor(\
			std::shared_ptr< dolfin::TYPE > *), \
				  0, &newmem);
      if (SWIG_IsOK(res))
      {
        if (itemp)
        {
          tempshared = *(reinterpret_cast< std::shared_ptr<dolfin::TYPE> * >(itemp));
	  tmp_vec.push_back(std::make_pair(value, tempshared.get()));
        }

        // If we need to release memory
        if (newmem & SWIG_CAST_NEW_MEMORY)
          delete reinterpret_cast< std::shared_ptr< dolfin::TYPE > * >(itemp);
      }
      else
        SWIG_exception(SWIG_TypeError, "list of tuples of float and TYPE expected. (Bad conversion)");
    }
  }

  $1 = tmp_vec;
}
%enddef

//-----------------------------------------------------------------------------
// Instantiate typemap
//-----------------------------------------------------------------------------
IN_TYPEMAPS_STD_VECTOR_OF_PAIRS_OF_DOUBLE_AND_POINTER(Function)

//-----------------------------------------------------------------------------
// Instantiate Hierarchical FunctionSpace, Function template class
//-----------------------------------------------------------------------------
%template (HierarchicalFunctionSpace) dolfin::Hierarchical<dolfin::FunctionSpace>;
%template (HierarchicalFunction) dolfin::Hierarchical<dolfin::Function>;
