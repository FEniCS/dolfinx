// Copyright (C) 2018 Chris N. Richardson and Garth N. Wells
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include <dolfinx/mesh/Mesh.h>
#include <dolfinx/mesh/MeshTags.h>
#include <dolfinx/refinement/plaza.h>
#include <dolfinx/refinement/refine.h>
#include <dolfinx/refinement/utils.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

namespace dolfinx_wrappers
{

void refinement(py::module& m)
{

  py::enum_<dolfinx::refinement::plaza::RefinementOptions>(m,
                                                           "RefinementOptions")
      .value("parent_facet",
             dolfinx::refinement::plaza::RefinementOptions::parent_facet)
      .value("parent_cell",
             dolfinx::refinement::plaza::RefinementOptions::parent_cell)
      .value(
          "parent_cell_and_facet",
          dolfinx::refinement::plaza::RefinementOptions::parent_cell_and_facet);

  // dolfinx::refinement::refine
  m.def("refine",
        py::overload_cast<const dolfinx::mesh::Mesh&, bool>(
            &dolfinx::refinement::refine),
        py::arg("mesh"), py::arg("redistribute") = true);

  m.def("plaza_refine_data",
        py::overload_cast<const dolfinx::mesh::Mesh&, bool,
                          dolfinx::refinement::plaza::RefinementOptions>(
            &dolfinx::refinement::plaza::refine),
        py::arg("mesh"), py::arg("redistribute"), py::arg("options"));

  m.def("transfer_facet_meshtag", &dolfinx::refinement::transfer_facet_meshtag,
        py::arg("parent_meshtag"), py::arg("refined_mesh"),
        py::arg("parent_cell"), py::arg("parent_facet"));
  m.def("transfer_cell_meshtag", &dolfinx::refinement::transfer_cell_meshtag,
        py::arg("parent_meshtag"), py::arg("refined_mesh"),
        py::arg("parent_cell"));

  m.def(
      "refine",
      [](const dolfinx::mesh::Mesh& mesh,
         const py::array_t<std::int32_t, py::array::c_style>& edges,
         bool redistribute)
      {
        assert(edges.ndim() == 1);
        return dolfinx::refinement::refine(
            mesh, std::span(edges.data(), edges.size()), redistribute);
      },
      py::arg("mesh"), py::arg("edges"), py::arg("redistribute") = true);
}

} // namespace dolfinx_wrappers
