// Copyright (C) 2017 Chris Richardson and Garth N. Wells
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include <dolfin/math/basic.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace dolfin_wrappers
{
void math(py::module& m)
{
  // dolfin/math free functions
  m.def("ipow", &dolfin::ipow);
  m.def("near", &dolfin::near, py::arg("x0"), py::arg("x1"),
        py::arg("eps") = DOLFIN_EPS);
}
}
