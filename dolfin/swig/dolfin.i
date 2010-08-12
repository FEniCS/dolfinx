/* -*- C -*- */
// Copyright (C) 2005-2006 Johan Jansson
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Anders logg, 2005-2009.
// Modified by Ola Skavhaug, 2007-2009.
// Modified by Kent-Andre Mardal, 2008-2009.
// Modified by Johan Hake, 2008-2009.
// Modified by Garth N. Wells, 2009.
//
// First added:  2005-10-24
// Last changed: 2009-11-27

// The PyDOLFIN extension module
%module(package="dolfin", directors="1") cpp

%{
#include <dolfin/dolfin.h>
#define PY_ARRAY_UNIQUE_SYMBOL PyDOLFIN
#include <numpy/arrayobject.h>
%}

%init%{
import_array();
%}

// Global shared ptr declarations
%include "dolfin/swig/shared_ptr_classes.i"

// Global typemaps
%include "dolfin/swig/typemaps.i"
%include "dolfin/swig/numpy_typemaps.i"
%include "dolfin/swig/array_typemaps.i"
%include "dolfin/swig/std_vector_typemaps.i"
%include "dolfin/swig/std_set_typemaps.i"

// Global exceptions
%include <exception.i>
%include "dolfin/swig/exceptions.i"

// STL SWIG string class
%include <std_string.i>

// Turn on SWIG generated signature documentation and include doxygen
// generated docstrings (Need to run generate.py to update the latter)
%feature("autodoc", "1");
%include "dolfin/swig/docstrings.i"

// DOLFIN interface (Need to run generate.py to update this file)
%include "dolfin/swig/kernel_modules.i"

// Include information about swig version
%include "dolfin/swig/swig_version.i"
%include "dolfin/swig/defines.i"

