// Copyright (C) 2018 Chris N. Richardson and Garth N. Wells
//
// This file is part of DOLFINX (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include <dolfinx/mesh/Mesh.h>
#include <dolfinx/mesh/MeshFunction.h>
#include <dolfinx/refinement/refine.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace dolfinx_wrappers
{

void refinement(py::module& m)
{

  // dolfinx::refinement::refine
  m.def("refine",
        py::overload_cast<const dolfinx::mesh::Mesh&, bool>(
            &dolfinx::refinement::refine),
        py::arg("mesh"), py::arg("redistribute") = true);

  m.def("refine",
        py::overload_cast<const dolfinx::mesh::Mesh&,
                          const dolfinx::mesh::MeshFunction<int>&, bool>(
            &dolfinx::refinement::refine),
        py::arg("mesh"), py::arg("marker"), py::arg("redistribute") = true);
}

} // namespace dolfinx_wrappers
