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
// Last changed: 2009-09-07

// The PyDOLFIN extension module
%module(package="dolfin", directors="1") cpp

%{
#define protected public
#include <dolfin/dolfin.h>
#include <dolfin/common/NoDeleter.h>
#define PY_ARRAY_UNIQUE_SYMBOL PyDolfin
#include <numpy/arrayobject.h>
using namespace dolfin;
%}

%init%{
import_array();
%}

// Global shared ptr declarations
%include "dolfin/swig/shared_ptr_classes.i"

// Global renames
%include "dolfin/swig/renames.i"

// Global ignores
%include "dolfin/swig/ignores.i"

// Global typemaps
%include "dolfin/swig/typemaps.i"
%include "dolfin/swig/numpy_typemaps.i"
%include "dolfin/swig/std_vector_typemaps.i"

// Global directors
%include "dolfin/swig/directors.i"

// Global exceptions
%include <exception.i>
%include "dolfin/swig/exceptions.i"

// STL SWIG string class
%include <std_string.i>

// Include doxygen generated docstrings and turn on SWIG generated signature documentation
%include "dolfin/swig/docstrings.i"
%feature("autodoc", "1");

// Import the dolfin::uint type
%import  "dolfin/common/types.h"

// DOLFIN interface
%include "dolfin/swig/headers.i"

// Include information about swig version
%include "dolfin/swig/swig_version.i"
%include "dolfin/swig/defines.i"

