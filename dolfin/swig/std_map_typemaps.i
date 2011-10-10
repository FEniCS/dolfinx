/* -*- C -*- */
// Copyright (C) 2011 Johan Hake
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
// First added:  2011-09-27
// Last changed: 2011-10-09

//=============================================================================
// In this file we declare what types that should be able to be passed using
// some kind of map typemap.
//=============================================================================

//-----------------------------------------------------------------------------
// Declare a dummy map class
// This makes SWIG aware of the template type
//-----------------------------------------------------------------------------
namespace std
{
  template <typename T0, typename T1> class map
  {
  };
}

namespace boost
{
  template <typename T0, typename T1> class unordered_map
  {
  };
}

//-----------------------------------------------------------------------------
// Macro for defining out typemaps for various maps of primitives
//
//    const std::map<KEY_TYPE, VALUE_TYPE>&, 
//    const boost::unordered_map<KEY_TYPE, VALUE_TYPE>&
//    const std::map<KEY_TYPE, std::vector<VALUE_TYPE> >&
//
// KEY_TYPE   : The key type
// VALUE_TYPE : The value type
// TYPENAME   : The name of the type (used to construct a NumPy array)
//-----------------------------------------------------------------------------
%define MAP_OUT_TYPEMAPS(KEY_TYPE, VALUE_TYPE, TYPENAME)
%typemap(out) const boost::unordered_map<KEY_TYPE, VALUE_TYPE>& 
 (boost::unordered_map<KEY_TYPE, VALUE_TYPE>::const_iterator it, 
  PyObject* item0, PyObject* item1)
{
  // boost::unordered_map<KEY_TYPE, VALUE_TYPE>
  $result = PyDict_New();
  for (it=$1->begin(); it!=$1->end(); ++it){
    item0 = PyInt_FromLong(static_cast<long>(it->first));
    item1 = PyInt_FromLong(static_cast<long>(it->second));
    PyDict_SetItem($result, item0, item1);
    Py_XDECREF(item0);    
    Py_XDECREF(item1);    
  }
  
}

%typemap(out) const std::map<KEY_TYPE, VALUE_TYPE>& \
 (std::map<KEY_TYPE, VALUE_TYPE>::const_iterator it, 
  PyObject* item0, PyObject* item1)
{
  // std::map<KEY_TYPE, VALUE_TYPE>
  $result = PyDict_New();
  for (it=$1->begin(); it!=$1->end(); ++it){
    item0 = PyInt_FromLong(static_cast<long>(it->first));
    item1 = PyInt_FromLong(static_cast<long>(it->second));
    PyDict_SetItem($result, item0, item1);
    Py_XDECREF(item0);    
    Py_XDECREF(item1);    
  }
}

%typemap(out) const std::map<KEY_TYPE, std::vector<VALUE_TYPE> >& \
 (std::map<KEY_TYPE, std::vector<VALUE_TYPE> >::const_iterator it, 
  PyObject* item0, PyObject* item1)
{
  // std::map<KEY_TYPE, std::vector<VALUE_TYPE> >
  $result = PyDict_New();
  for (it=$1->begin(); it!=$1->end(); ++it){
    item0 = PyInt_FromLong(static_cast<long>(it->first));
    item1 = %make_numpy_array(1, TYPENAME)(it->second.size(), &it->second[0], false);
    PyDict_SetItem($result, item0, item1);
    Py_XDECREF(item0);    
    Py_XDECREF(item1);    
  }
}

%enddef

//-----------------------------------------------------------------------------
// Run the macro and instantiate the typemaps
//-----------------------------------------------------------------------------
MAP_OUT_TYPEMAPS(unsigned int, unsigned int, uint)
