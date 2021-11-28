// Copyright (C) 2018 Chris N. Richardson and Garth N. Wells
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include <dolfinx/mesh/Mesh.h>
#include <dolfinx/refinement/refine.h>
#include <pybind11/numpy.h>
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

  m.def(
      "refine",
      [](const dolfinx::mesh::Mesh& mesh,
         const py::array_t<std::int32_t, py::array::c_style>& edges,
         bool redistribute)
      {
        assert(edges.ndim() == 1);
        return dolfinx::refinement::refine(
            mesh, xtl::span(edges.data(), edges.size()), redistribute);
      },
      py::arg("mesh"), py::arg("edges"), py::arg("redistribute") = true);
}

} // namespace dolfinx_wrappers
