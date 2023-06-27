// Copyright (C) 2018 Chris N. Richardson and Garth N. Wells
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "array.h"
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
  py::enum_<dolfinx::refinement::plaza::Option>(m, "RefinementOption")
      .value("none", dolfinx::refinement::plaza::Option::none)
      .value("parent_facet", dolfinx::refinement::plaza::Option::parent_facet)
      .value("parent_cell", dolfinx::refinement::plaza::Option::parent_cell)
      .value("parent_cell_and_facet",
             dolfinx::refinement::plaza::Option::parent_cell_and_facet);

  // dolfinx::refinement::refine
  m.def("refine",
        py::overload_cast<const dolfinx::mesh::Mesh<float>&, bool>(
            &dolfinx::refinement::refine<float>),
        py::arg("mesh"), py::arg("redistribute") = true);
  m.def("refine",
        py::overload_cast<const dolfinx::mesh::Mesh<double>&, bool>(
            &dolfinx::refinement::refine<double>),
        py::arg("mesh"), py::arg("redistribute") = true);
  m.def(
      "refine",
      [](const dolfinx::mesh::Mesh<float>& mesh,
         const py::array_t<std::int32_t, py::array::c_style>& edges,
         bool redistribute)
      {
        assert(edges.ndim() == 1);
        return dolfinx::refinement::refine(
            mesh, std::span(edges.data(), edges.size()), redistribute);
      },
      py::arg("mesh"), py::arg("edges"), py::arg("redistribute") = true);
  m.def(
      "refine",
      [](const dolfinx::mesh::Mesh<double>& mesh,
         const py::array_t<std::int32_t, py::array::c_style>& edges,
         bool redistribute)
      {
        assert(edges.ndim() == 1);
        return dolfinx::refinement::refine(
            mesh, std::span(edges.data(), edges.size()), redistribute);
      },
      py::arg("mesh"), py::arg("edges"), py::arg("redistribute") = true);
  m.def(
      "refine_plaza",
      [](const dolfinx::mesh::Mesh<float>& mesh0, bool redistribute,
         dolfinx::refinement::plaza::Option option)
      {
        auto [mesh1, cell, facet]
            = dolfinx::refinement::plaza::refine(mesh0, redistribute, option);
        return std::tuple{std::move(mesh1), as_pyarray(std::move(cell)),
                          as_pyarray(std::move(facet))};
      },
      py::arg("mesh"), py::arg("redistribute"), py::arg("option"));
  m.def(
      "refine_plaza",
      [](const dolfinx::mesh::Mesh<double>& mesh0, bool redistribute,
         dolfinx::refinement::plaza::Option option)
      {
        auto [mesh1, cell, facet]
            = dolfinx::refinement::plaza::refine(mesh0, redistribute, option);
        return std::tuple{std::move(mesh1), as_pyarray(std::move(cell)),
                          as_pyarray(std::move(facet))};
      },
      py::arg("mesh"), py::arg("redistribute"), py::arg("option"));
  m.def(
      "refine_plaza",
      [](const dolfinx::mesh::Mesh<float>& mesh0,
         py::array_t<std::int32_t> edges, bool redistribute,
         dolfinx::refinement::plaza::Option option)
      {
        assert(edges.ndim() == 1);
        auto [mesh1, cell, facet] = dolfinx::refinement::plaza::refine(
            mesh0, std::span<const std::int32_t>(edges.data(), edges.size()),
            redistribute, option);
        return std::tuple{std::move(mesh1), as_pyarray(std::move(cell)),
                          as_pyarray(std::move(facet))};
      },
      py::arg("mesh"), py::arg("edges"), py::arg("redistribute"),
      py::arg("option"));
  m.def(
      "refine_plaza",
      [](const dolfinx::mesh::Mesh<double>& mesh0,
         py::array_t<std::int32_t> edges, bool redistribute,
         dolfinx::refinement::plaza::Option option)
      {
        assert(edges.ndim() == 1);
        auto [mesh1, cell, facet] = dolfinx::refinement::plaza::refine(
            mesh0, std::span<const std::int32_t>(edges.data(), edges.size()),
            redistribute, option);
        return std::tuple{std::move(mesh1), as_pyarray(std::move(cell)),
                          as_pyarray(std::move(facet))};
      },
      py::arg("mesh"), py::arg("edges"), py::arg("redistribute"),
      py::arg("option"));

  m.def(
      "transfer_facet_meshtag",
      [](const dolfinx::mesh::MeshTags<std::int32_t>& parent_meshtag,
         std::shared_ptr<const dolfinx::mesh::Topology> topology1,
         const py::array_t<std::int32_t, py::array::c_style>& parent_cell,
         const py::array_t<std::int8_t, py::array::c_style>& parent_facet)
      {
        int tdim = parent_meshtag.topology()->dim();
        if (parent_meshtag.dim() != tdim - 1)
          throw std::runtime_error("Input meshtag is not facet-based");
        auto [entities, values] = dolfinx::refinement::transfer_facet_meshtag(
            parent_meshtag, *topology1,
            std::span<const std::int32_t>(parent_cell.data(),
                                          parent_cell.size()),
            std::span<const std::int8_t>(parent_facet.data(),
                                         parent_facet.size()));
        return dolfinx::mesh::MeshTags<std::int32_t>(
            topology1, tdim - 1, std::move(entities), std::move(values));
      },
      py::arg("parent_meshtag"), py::arg("refined_mesh"),
      py::arg("parent_cell"), py::arg("parent_facet"));
  m.def(
      "transfer_cell_meshtag",
      [](const dolfinx::mesh::MeshTags<std::int32_t>& parent_meshtag,
         std::shared_ptr<const dolfinx::mesh::Topology> topology1,
         const py::array_t<std::int32_t, py::array::c_style>& parent_cell)
      {
        int tdim = parent_meshtag.topology()->dim();
        if (parent_meshtag.dim() != tdim)
          throw std::runtime_error("Input meshtag is not cell-based");
        if (parent_meshtag.topology()->index_map(tdim)->num_ghosts() > 0)
          throw std::runtime_error("Ghosted meshes are not supported");
        auto [entities, values] = dolfinx::refinement::transfer_cell_meshtag(
            parent_meshtag, *topology1,
            std::span<const std::int32_t>(parent_cell.data(),
                                          parent_cell.size()));
        return dolfinx::mesh::MeshTags<std::int32_t>(
            topology1, tdim, std::move(entities), std::move(values));
      },
      py::arg("parent_meshtag"), py::arg("refined_mesh"),
      py::arg("parent_cell"));
}

} // namespace dolfinx_wrappers
