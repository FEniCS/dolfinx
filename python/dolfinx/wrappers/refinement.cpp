// Copyright (C) 2018 Chris N. Richardson and Garth N. Wells
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include <dolfinx/mesh/Mesh.h>
#include <dolfinx/mesh/MeshTags.h>
#include <dolfinx/refinement/parent_map.h>
#include <dolfinx/refinement/refine.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

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
                          const dolfinx::mesh::MeshTags<std::int8_t>&, bool>(
            &dolfinx::refinement::refine),
        py::arg("mesh"), py::arg("marker"), py::arg("redistribute") = true);

  // dolfinx::refinement::ParentRelationshipInfo
  py::class_<dolfinx::refinement::ParentRelationshipInfo,
             std::shared_ptr<dolfinx::refinement::ParentRelationshipInfo>>(
      m, "ParentRelationshipInfo", py::dynamic_attr(),
      "ParentRelationshipInfo object")
      .def_property_readonly(
          "parent_map",
          &dolfinx::refinement::ParentRelationshipInfo::parent_map,
          "The map from the vertices to the entities of the parent mesh.");
}

} // namespace dolfinx_wrappers
