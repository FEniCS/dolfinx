// Copyright (C) 2017-2021 Chris Richardson and Garth N. Wells
//
// This file is part of DOLFINX (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include <adios2.h>
#include <pybind11/pybind11.h>

// pybind11 casters for ADIOS C++/ADIOS Python objects

namespace py = pybind11;

namespace py
{
namespace detail
{

template <>
class type_caster<adios2::ADIOS>
{
public:
  {
    /**
     * This macro establishes the name 'inty' in
     * function signatures and declares a local variable
     * 'value' of type inty
     */
    PYBIND11_TYPE_CASTER(inty, _("inty"));

    /**
     * Conversion part 1 (Python->C++): convert a PyObject into a inty
     * instance or return false upon failure. The second argument
     * indicates whether implicit conversions should be applied.
     */
    bool load(handle src, bool)
    {
      /* Extract PyObject from handle */
      PyObject* source = src.ptr();
      /* Try converting into a Python integer value */
      PyObject* tmp = PyNumber_Long(source);
      if (!tmp)
        return false;
      /* Now try to convert into a C++ int */
      value.long_value = PyLong_AsLong(tmp);
      Py_DECREF(tmp);
      /* Ensure return code was OK (to avoid out-of-range errors etc) */
      return !(value.long_value == -1 && !PyErr_Occurred());
    }

    /**
     * Conversion part 2 (C++ -> Python): convert an inty instance into
     * a Python object. The second and third arguments are used to
     * indicate the return value policy and parent object (for
     * ``return_value_policy::reference_internal``) and are generally
     * ignored by implicit casters.
     */
    static handle cast(inty src, return_value_policy /* policy */,
                       handle /* parent */)
    {
      return PyLong_FromLong(src.long_value);
    }
  };
} // namespace detail
} // namespace py
