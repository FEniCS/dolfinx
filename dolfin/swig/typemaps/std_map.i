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
// Last changed: 2013-02-08

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
// Help macro for defining (arg)out typemaps for either boost::unordered_map or
// std::map
//
//    const MAP_TYPE<KEY_TYPE, VALUE_TYPE>&, (out)
//    const MAP_TYPE<KEY_TYPE, std::vector<VALUE_TYPE> >& (out)
//    MAP_TYPE<uint, VALUE_TYPE>& (argout)
//
//-----------------------------------------------------------------------------
%define MAP_SPECIFIC_OUT_TYPEMAPS(MAP_TYPE, KEY_TYPE, VALUE_TYPE, TYPENAME)

%typemap(out) const MAP_TYPE<KEY_TYPE, VALUE_TYPE>&
 (MAP_TYPE<KEY_TYPE, VALUE_TYPE>::const_iterator it,
  PyObject* item0, PyObject* item1)
{
  // const MAP_TYPE<KEY_TYPE, VALUE_TYPE>& (out)
  $result = PyDict_New();
  for (it=$1->begin(); it!=$1->end(); ++it){
    item0 = SWIG_From_dec(KEY_TYPE)(it->first);
    item1 = SWIG_From_dec(VALUE_TYPE)(it->second);
    PyDict_SetItem($result, item0, item1);
    Py_XDECREF(item0);
    Py_XDECREF(item1);
  }
}

%typemap(out) MAP_TYPE<std::pair<KEY_TYPE, KEY_TYPE>, VALUE_TYPE>&
 (MAP_TYPE<std::pair<KEY_TYPE, KEY_TYPE>, VALUE_TYPE>::const_iterator it,
  PyObject* item0, PyObject* item1, PyObject* item2, PyObject* item3)
{
  // MAP_TYPE<std::pair<KEY_TYPE, KEY_TYPE>, VALUE_TYPE>& (out)
  $result = PyDict_New();
  for (it=$1->begin(); it!=$1->end(); ++it){
    //item0 = SWIG_From_dec(KEY_TYPE)(it->first.first);
    //item1 = SWIG_From_dec(KEY_TYPE)(it->first.second);
    item2 = Py_BuildValue("ii", it->first.first, it->first.second);
    item3 = SWIG_From_dec(VALUE_TYPE)(it->second);
    
    PyDict_SetItem($result, item2, item3);
    //Py_XDECREF(item0);
    //Py_XDECREF(item1);
    Py_XDECREF(item2);
    Py_XDECREF(item3);
  }
}

%typemap(out) const MAP_TYPE<KEY_TYPE, std::vector<VALUE_TYPE> >& \
 (MAP_TYPE<KEY_TYPE, std::vector<VALUE_TYPE> >::const_iterator it,
  PyObject* item0, PyObject* item1)
{
  // const MAP_TYPE<KEY_TYPE, std::vector<VALUE_TYPE> > (out)
  $result = PyDict_New();
  for (it=$1->begin(); it!=$1->end(); ++it){
    item0 = SWIG_From_dec(KEY_TYPE)(it->first);
    item1 = %make_numpy_array(1, TYPENAME)(it->second.size(), &it->second[0], false);
    PyDict_SetItem($result, item0, item1);
    Py_XDECREF(item0);
    Py_XDECREF(item1);
  }
}

%typemap (in, numinputs=0) MAP_TYPE<KEY_TYPE, VALUE_TYPE>&
  (MAP_TYPE<KEY_TYPE, VALUE_TYPE> map_temp)
{
  // const MAP_TYPE<KEY_TYPE, VALUE_TYPE>& (argout)
  $1 = &map_temp;
}

%typemap(argout) MAP_TYPE<KEY_TYPE, VALUE_TYPE>&
  (MAP_TYPE<KEY_TYPE, VALUE_TYPE>::const_iterator it,
   PyObject* item0, PyObject* item1)
{
  // const MAP_TYPE<KEY_TYPE, VALUE_TYPE>& (argout)
  PyObject *ret = PyDict_New();
  for (it=$1->begin(); it!=$1->end(); ++it){
    item0 = SWIG_From_dec(KEY_TYPE)(it->first);
    item1 = SWIG_From_dec(VALUE_TYPE)(it->second);
    PyDict_SetItem(ret, item0, item1);
    Py_XDECREF(item0);
    Py_XDECREF(item1);
  }

  // Append the output to $result
  %append_output(ret);
}
%enddef

//-----------------------------------------------------------------------------
// Macro for defining out typemaps for various maps of primitives
//
// KEY_TYPE   : The key type
// VALUE_TYPE : The value type
// TYPENAME   : The name of the type (used to construct a NumPy array)
//-----------------------------------------------------------------------------
%define MAP_OUT_TYPEMAPS(KEY_TYPE, VALUE_TYPE, TYPENAME)
MAP_SPECIFIC_OUT_TYPEMAPS(boost::unordered_map, KEY_TYPE, VALUE_TYPE, TYPENAME)
MAP_SPECIFIC_OUT_TYPEMAPS(std::map, KEY_TYPE, VALUE_TYPE, TYPENAME)
%enddef

//-----------------------------------------------------------------------------
// Run the macro and instantiate the typemaps
//-----------------------------------------------------------------------------
// NOTE: SWIG BUG
// NOTE: Because of bug introduced by SWIG 2.0.5 we cannot use templated versions
// NOTE: of typdefs, which means we need to use unsigned int instead of dolfin::uint
// NOTE: in typemaps
// NOTE: Well... to get std::size_t up and running we need to use typedefs.
MAP_OUT_TYPEMAPS(unsigned int, unsigned int, uint)
MAP_OUT_TYPEMAPS(std::size_t, double, double)
MAP_OUT_TYPEMAPS(std::size_t, std::size_t, size_t)
