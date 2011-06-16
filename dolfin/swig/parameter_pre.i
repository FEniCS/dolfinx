/* -*- C -*- */
// Copyright (C) 2006-2009 Johan Hake
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
// First added:  2009-05-12
// Last changed: 2009-10-12
//
// ===========================================================================
// SWIG directives for the DOLFIN parameter kernel module (pre)
//
// The directives in this file are applied _before_ the header files of the
// modules has been loaded.
// ===========================================================================

// ---------------------------------------------------------------------------
// Renames and ignores for Parameter
// For some obscure reason we need to rename Parameter
// ---------------------------------------------------------------------------
%rename (ParameterValue) dolfin::Parameter;
%rename (__nonzero__) dolfin::Parameter::operator bool;
%rename (__int__) dolfin::Parameter::operator int;
%rename (__float__) dolfin::Parameter::operator double;
%rename (__str__) dolfin::Parameter::operator std::string;
%rename (_assign) dolfin::Parameter::operator=;
%rename (_get_int_range) dolfin::Parameter::get_range(int& min_value, int& max_value) const;
%rename (_get_real_range) dolfin::Parameter::get_range(double& min_value, double& max_value) const;
%rename (_get_string_range) dolfin::Parameter::get_range(std::set<std::string>& range) const;
%ignore dolfin::Parameter::operator dolfin::uint;

// ---------------------------------------------------------------------------
// Renames and ignores for Parameters
// ---------------------------------------------------------------------------
%rename (_assign_bool) dolfin::Parameter::operator= (bool value);
%rename (_add) dolfin::Parameters::add;
%rename (_add_bool) dolfin::Parameters::add(std::string key, bool value);
%rename (_get_parameter_keys) dolfin::Parameters::get_parameter_keys;
%rename (_get_parameter_set_keys) dolfin::Parameters::get_parameter_set_keys;
%rename (_get_parameter_set) dolfin::Parameters::operator();
%rename (_get_parameter) dolfin::Parameters::operator[];
%rename (assign) dolfin::Parameters::operator=;
%ignore dolfin::Parameters::parse;
%ignore dolfin::Parameters::update;

// ---------------------------------------------------------------------------
// Typemaps (in) for std::set<std::string>
// ---------------------------------------------------------------------------
%typecheck(SWIG_TYPECHECK_STRING_ARRAY) std::set<std::string> {
    $1 = PySequence_Check($input) ? 1 : 0;
}

%typemap(in) std::set<std::string> (std::set<std::string> tmp) {
  int i;
  if (!PyList_Check($input)) {
    PyErr_SetString(PyExc_ValueError,"expected a list of 'str'");
    return NULL;
  }
  int list_length = PyList_Size($input);
  if (!list_length > 0){
    PyErr_SetString(PyExc_ValueError,"expected a list with length > 0");
    return NULL;
  }
  for (i = 0; i < list_length; i++) {
    PyObject *o = PyList_GetItem($input,i);
    if (PyString_Check(o)) {
      tmp.insert(std::string(PyString_AsString(o)));
    } else {
      PyErr_SetString(PyExc_TypeError,"provide a list of strings");
      return NULL;
    }
  }
  $1 = tmp;
}

// ---------------------------------------------------------------------------
// Typemaps (argout) for std::vector<std::string>&
// ---------------------------------------------------------------------------
%typemap(in, numinputs=0) std::vector<std::string>& keys (std::vector<std::string> tmp_vec){
  $1 = &tmp_vec;
}

%typemap(argout) std::vector<std::string>& keys
{
  int size = $1->size();
  PyObject* ret = PyList_New(size);
  PyObject* tmp_Py_str = 0;
  for (int i=0; i < size; i++)
  {
    tmp_Py_str = PyString_FromString((*$1)[i].c_str());
    if (PyList_SetItem(ret,i,tmp_Py_str)<0)
    {
      PyErr_SetString(PyExc_ValueError,"something wrong happened when copying std::string to Python");
      return NULL;
    }
  }
  $result = ret;
}

// ---------------------------------------------------------------------------
// Typemaps (argout) for int &min_value, int &max_value
// ---------------------------------------------------------------------------
%typemap(in, numinputs=0) (int &min_value, int &max_value) (int min_temp, int max_temp){
  $1 = &min_temp; $2 = &max_temp;
}

%typemap(argout) (int &min_value, int &max_value)
{
  $result = Py_BuildValue("ii", *$1, *$2);
}

// ---------------------------------------------------------------------------
// Typemaps (argout) for real &min_value, real &max_value
// ---------------------------------------------------------------------------
//%typemap(in, numinputs=0) (dolfin::real &min_value, dolfin::real &max_value) ( dolfin::real min_temp, dolfin::real max_temp){
//  $1 = &min_temp; $2 = &max_temp;
//}

/*
%typemap(argout) (dolfin::real &min_value, dolfin::real &max_value)
{
  #ifdef HAS_GMP
  $result = Py_BuildValue("dd", $1->get_d(), $2->get_d());
  #else
  $result = Py_BuildValue("dd", *$1, *$2);
  #endif
}
*/

// ---------------------------------------------------------------------------
// Typemaps (argout) for std::set<std::string>&
// ---------------------------------------------------------------------------
%typemap(in, numinputs=0) std::set<std::string>& range (std::set<std::string> tmp_set){
  $1 = &tmp_set;
}

%typemap(argout) std::set<std::string>& range
{
  int size = $1->size();
  PyObject* ret = PyList_New(size);
  PyObject* tmp_Py_str = 0;
  std::set<std::string>::iterator it;
  int i = 0;
  for ( it=$1->begin() ; it != $1->end(); it++ )
  {
    tmp_Py_str = PyString_FromString(it->c_str());
    if (PyList_SetItem(ret, i, tmp_Py_str)<0)
    {
      PyErr_SetString(PyExc_ValueError,"something wrong happened when copying std::string to Python");
      return NULL;
    }
    i++;
  }
  $result = ret;
}

