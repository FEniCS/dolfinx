// Copyright (C) 2018-2024 Chris N. Richardson, Garth N. Wells and Paul T.
// Kühner
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "dolfinx_wrappers/MPICommWrapper.h"
#include "dolfinx_wrappers/array.h"
#include "dolfinx_wrappers/mesh.h"
#include <concepts>
#include <cstdint>
#include <dolfinx/mesh/Mesh.h>
#include <dolfinx/mesh/MeshTags.h>
#include <dolfinx/refinement/interval.h>
#include <dolfinx/refinement/option.h>
#include <dolfinx/refinement/refine.h>
#include <dolfinx/refinement/uniform.h>
#include <dolfinx/refinement/utils.h>
#include <functional>
#include <memory>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/function.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/variant.h>
#include <nanobind/stl/vector.h>
#include <optional>
#include <span>
#include <stdexcept>
#include <variant>

namespace nb = nanobind;

namespace
{
template <std::floating_point T>
void export_refinement(nb::module_& m)
{
  m.def(
      "uniform_refine", [](const dolfinx::mesh::Mesh<T>& mesh)
      { return dolfinx::refinement::uniform_refine<T>(mesh); },
      nb::arg("mesh"));

  m.def(
      "refine",
      [](const dolfinx::mesh::Mesh<T>& mesh,
         std::optional<
             nb::ndarray<const std::int32_t, nb::ndim<1>, nb::c_contig>>
             edges,
         std::variant<dolfinx::refinement::IdentityPartitionerPlaceholder,
                      std::optional<dolfinx_wrappers::part::impl::
                                        PythonCellPartitionFunction>>
             partitioner,
         dolfinx::refinement::Option option)
      {
        std::optional<std::span<const std::int32_t>> cpp_edges(std::nullopt);
        if (edges.has_value())
        {
          cpp_edges.emplace(
              std::span(edges.value().data(), edges.value().size()));
        }

        std::variant<dolfinx::refinement::IdentityPartitionerPlaceholder,
                     dolfinx_wrappers::part::impl::CppCellPartitionFunction>
            cpp_partitioner
            = dolfinx::refinement::IdentityPartitionerPlaceholder();
        if (std::holds_alternative<std::optional<
                dolfinx_wrappers::part::impl::PythonCellPartitionFunction>>(
                partitioner))
        {
          auto optional = std::get<std::optional<
              dolfinx_wrappers::part::impl::PythonCellPartitionFunction>>(
              partitioner);
          if (!optional.has_value())
            cpp_partitioner
                = dolfinx_wrappers::part::impl::CppCellPartitionFunction(
                    nullptr);
          else
          {
            cpp_partitioner
                = dolfinx_wrappers::part::impl::create_cell_partitioner_cpp(
                    optional.value());
          }
        }

        auto [mesh1, cell, facet] = dolfinx::refinement::refine(
            mesh, cpp_edges, cpp_partitioner, option);

        std::optional<nb::ndarray<std::int32_t, nb::numpy>> python_cell(
            std::nullopt);
        if (cell.has_value())
        {
          python_cell.emplace(
              dolfinx_wrappers::as_nbarray(std::move(cell.value())));
        }

        std::optional<nb::ndarray<std::int8_t, nb::numpy>> python_facet(
            std::nullopt);
        if (facet.has_value())
        {
          python_facet.emplace(
              dolfinx_wrappers::as_nbarray(std::move(facet.value())));
        }

        return std::tuple{std::move(mesh1), std::move(python_cell),
                          std::move(python_facet)};
      },
      nb::arg("mesh"), nb::arg("edges").none(), nb::arg("partitioner").none(),
      nb::arg("option"));
}
} // namespace

namespace dolfinx_wrappers
{

void refinement(nb::module_& m)
{
  export_refinement<float>(m);
  export_refinement<double>(m);

  nb::class_<dolfinx::refinement::IdentityPartitionerPlaceholder>(
      m, "IdentityPartitionerPlaceholder")
      .def(nb::init<>());

  nb::enum_<dolfinx::refinement::Option>(m, "RefinementOption")
      .value("none", dolfinx::refinement::Option::none)
      .value("parent_facet", dolfinx::refinement::Option::parent_facet)
      .value("parent_cell", dolfinx::refinement::Option::parent_cell)
      .value("parent_cell_and_facet",
             dolfinx::refinement::Option::parent_cell_and_facet);
  m.def(
      "transfer_facet_meshtag",
      [](const dolfinx::mesh::MeshTags<std::int32_t>& parent_meshtag,
         const std::shared_ptr<const dolfinx::mesh::Topology>& topology1,
         const nb::ndarray<const std::int32_t, nb::ndim<1>, nb::c_contig>&
             parent_cell,
         const nb::ndarray<const std::int8_t, nb::ndim<1>, nb::c_contig>&
             parent_facet)
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
         const std::shared_ptr<const dolfinx::mesh::Topology>& topology1,
         const nb::ndarray<const std::int32_t, nb::ndim<1>, nb::c_contig>&
             parent_cell)
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
