// Copyright (C) 2018-2024 Chris N. Richardson, Garth N. Wells and Paul T.
// Kühner
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include "array.h"
#include "mesh.h"
#include <concepts>
#include <dolfinx/mesh/Mesh.h>
#include <dolfinx/refinement/option.h>
#include <dolfinx/refinement/refine.h>
#include <dolfinx/refinement/uniform.h>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/function.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/variant.h>
#include <optional>
#include <span>
#include <variant>

namespace dolfinx_wrappers
{

template <std::floating_point T>
void declare_refinement(nanobind::module_& m)
{
  namespace nb = nanobind;

  m.def(
      "uniform_refine",
      [](const dolfinx::mesh::Mesh<T>& mesh,
         std::optional<
             dolfinx_wrappers::part::impl::PythonCellPartitionFunction>
             partitioner)
      {
        dolfinx_wrappers::part::impl::CppCellPartitionFunction cpp_partitioner;
        if (partitioner.has_value())
        {
          cpp_partitioner
              = dolfinx_wrappers::part::impl::create_cell_partitioner_cpp(
                  partitioner.value());
        }
        else
        {
          cpp_partitioner
              = dolfinx_wrappers::part::impl::CppCellPartitionFunction(nullptr);
        }
        return dolfinx::refinement::uniform_refine<T>(mesh, cpp_partitioner);
      },
      nb::arg("mesh"), nb::arg("partitioner").none());

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

} // namespace dolfinx_wrappers
