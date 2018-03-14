// Copyright (C) 2018 Chris N. Richardson and Garth N. Wells
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include <dolfin/mesh/Mesh.h>
#include <dolfin/refinement/refine.h>

#include "casters.h"

namespace py = pybind11;

namespace dolfin_wrappers {

void refinement(py::module &m) {

  // dolfin::refinement::refine
  m.def("refine", py::overload_cast<const dolfin::mesh::Mesh &, bool>(
                      &dolfin::refinement::refine),
        py::arg("mesh"), py::arg("redistribute") = true);
}

} // namespace dolfin_wrappers
