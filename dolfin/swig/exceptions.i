/* -*- C -*- */
// Copyright (C) 2007-2009 Ola Skavhaug
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
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with DOLFIN.  If not, see <http://www.gnu.org/licenses/>.
//
// Modified by Johan Hake, 2009.
//
// First added:  2007-05-15
// Last changed: 2010-04-23

// ===========================================================================
// SWIG directives for exception handling in PyDOLFIN
// ===========================================================================

// ---------------------------------------------------------------------------
// Function that handles exceptions. Reduces code bloat.
// ---------------------------------------------------------------------------
%{
SWIGINTERN void handle_dolfin_exceptions()
{
  // Re-throw any exception
  try {
    throw;
  }
  
  // all logic_error subclasses
  catch (std::logic_error &e) {
    PyErr_SetString(PyExc_StandardError, const_cast<char*>(e.what()));
  }

  // all runtime_error subclasses
  catch (std::runtime_error &e) {
    PyErr_SetString(PyExc_RuntimeError, const_cast<char*>(e.what()));
  }

  // all the rest
  catch (std::exception &e) {
    PyErr_SetString(PyExc_Exception, const_cast<char*>(e.what()));
  }

}
%}

// ---------------------------------------------------------------------------
// Define the code that each call to DOLFIN should be wrapped in
// ---------------------------------------------------------------------------
%exception {
  try {
    $action
  }
  catch (...){
    // No need to call PyErr_SetString if the error originates from Python
    if (!PyErr_Occurred()) {
      handle_dolfin_exceptions();
    }
    SWIG_fail;
  }
}


