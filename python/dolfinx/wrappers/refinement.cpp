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
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/vector.h>

namespace nb = nanobind;

namespace dolfinx_wrappers
{
void refinement(nb::module_& m)
{
  nb::enum_<dolfinx::refinement::plaza::Option>(m, "RefinementOption")
      .value("none", dolfinx::refinement::plaza::Option::none)
      .value("parent_facet", dolfinx::refinement::plaza::Option::parent_facet)
      .value("parent_cell", dolfinx::refinement::plaza::Option::parent_cell)
      .value("parent_cell_and_facet",
             dolfinx::refinement::plaza::Option::parent_cell_and_facet);

  // dolfinx::refinement::refine
  m.def("refine",
        nb::overload_cast<const dolfinx::mesh::Mesh<float>&, bool>(
            &dolfinx::refinement::refine<float>),
        nb::arg("mesh"), nb::arg("redistribute") = true);
  m.def("refine",
        nb::overload_cast<const dolfinx::mesh::Mesh<double>&, bool>(
            &dolfinx::refinement::refine<double>),
        nb::arg("mesh"), nb::arg("redistribute") = true);
  m.def(
      "refine",
      [](const dolfinx::mesh::Mesh<float>& mesh,
         nb::ndarray<const std::int32_t, nb::ndim<1>, nb::c_contig> edges,
         bool redistribute)
      {
        return dolfinx::refinement::refine(
            mesh, std::span(edges.data(), edges.size()), redistribute);
      },
      nb::arg("mesh"), nb::arg("edges"), nb::arg("redistribute") = true);
  m.def(
      "refine",
      [](const dolfinx::mesh::Mesh<double>& mesh,
         nb::ndarray<const std::int32_t, nb::ndim<1>, nb::c_contig> edges,
         bool redistribute)
      {
        return dolfinx::refinement::refine(
            mesh, std::span(edges.data(), edges.size()), redistribute);
      },
      nb::arg("mesh"), nb::arg("edges"), nb::arg("redistribute") = true);
  m.def(
      "refine_plaza",
      [](const dolfinx::mesh::Mesh<float>& mesh0, bool redistribute,
         dolfinx::refinement::plaza::Option option)
      {
        auto [mesh1, cell, facet]
            = dolfinx::refinement::plaza::refine(mesh0, redistribute, option);
        return std::tuple{std::move(mesh1), as_nbarray(std::move(cell)),
                          as_nbarray(std::move(facet))};
      },
      nb::arg("mesh"), nb::arg("redistribute"), nb::arg("option"));
  m.def(
      "refine_plaza",
      [](const dolfinx::mesh::Mesh<double>& mesh0, bool redistribute,
         dolfinx::refinement::plaza::Option option)
      {
        auto [mesh1, cell, facet]
            = dolfinx::refinement::plaza::refine(mesh0, redistribute, option);
        return std::tuple{std::move(mesh1), as_nbarray(std::move(cell)),
                          as_nbarray(std::move(facet))};
      },
      nb::arg("mesh"), nb::arg("redistribute"), nb::arg("option"));
  m.def(
      "refine_plaza",
      [](const dolfinx::mesh::Mesh<float>& mesh0,
         nb::ndarray<const std::int32_t, nb::ndim<1>, nb::c_contig> edges,
         bool redistribute, dolfinx::refinement::plaza::Option option)
      {
        auto [mesh1, cell, facet] = dolfinx::refinement::plaza::refine(
            mesh0, std::span<const std::int32_t>(edges.data(), edges.size()),
            redistribute, option);
        return std::tuple{std::move(mesh1), as_nbarray(std::move(cell)),
                          as_nbarray(std::move(facet))};
      },
      nb::arg("mesh"), nb::arg("edges"), nb::arg("redistribute"),
      nb::arg("option"));
  m.def(
      "refine_plaza",
      [](const dolfinx::mesh::Mesh<double>& mesh0,
         nb::ndarray<std::int32_t, nb::ndim<1>, nb::c_contig> edges,
         bool redistribute, dolfinx::refinement::plaza::Option option)
      {
        auto [mesh1, cell, facet] = dolfinx::refinement::plaza::refine(
            mesh0, std::span<const std::int32_t>(edges.data(), edges.size()),
            redistribute, option);
        return std::tuple{std::move(mesh1), as_nbarray(std::move(cell)),
                          as_nbarray(std::move(facet))};
      },
      nb::arg("mesh"), nb::arg("edges"), nb::arg("redistribute"),
      nb::arg("option"));

  m.def(
      "transfer_facet_meshtag",
      [](const dolfinx::mesh::MeshTags<std::int32_t>& parent_meshtag,
         std::shared_ptr<const dolfinx::mesh::Topology> topology1,
         nb::ndarray<const std::int32_t, nb::ndim<1>, nb::c_contig> parent_cell,
         nb::ndarray<const std::int8_t, nb::ndim<1>, nb::c_contig> parent_facet)
      {
        int tdim = parent_meshtag.topology()->dim();
        if (parent_meshtag.dim() != tdim - 1)
          throw std::runtime_error("Input meshtag is not facet-based");
        auto [entities, values] = dolfinx::refinement::transfer_facet_meshtag(
            parent_meshtag, *topology1,
            std::span(parent_cell.data(), parent_cell.size()),
            std::span(parent_facet.data(), parent_facet.size()));
        return dolfinx::mesh::MeshTags<std::int32_t>(
            topology1, tdim - 1, std::move(entities), std::move(values));
      },
      nb::arg("parent_meshtag"), nb::arg("refined_mesh"),
      nb::arg("parent_cell"), nb::arg("parent_facet"));
  m.def(
      "transfer_cell_meshtag",
      [](const dolfinx::mesh::MeshTags<std::int32_t>& parent_meshtag,
         std::shared_ptr<const dolfinx::mesh::Topology> topology1,
         nb::ndarray<const std::int32_t, nb::ndim<1>, nb::c_contig> parent_cell)
      {
        int tdim = parent_meshtag.topology()->dim();
        if (parent_meshtag.dim() != tdim)
          throw std::runtime_error("Input meshtag is not cell-based");
        if (parent_meshtag.topology()->index_map(tdim)->num_ghosts() > 0)
          throw std::runtime_error("Ghosted meshes are not supported");
        auto [entities, values] = dolfinx::refinement::transfer_cell_meshtag(
            parent_meshtag, *topology1,
            std::span(parent_cell.data(), parent_cell.size()));
        return dolfinx::mesh::MeshTags<std::int32_t>(
            topology1, tdim, std::move(entities), std::move(values));
      },
      nb::arg("parent_meshtag"), nb::arg("refined_mesh"),
      nb::arg("parent_cell"));
}

} // namespace dolfinx_wrappers
