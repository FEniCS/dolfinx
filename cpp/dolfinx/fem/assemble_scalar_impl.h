// Copyright (C) 2019-2020 Garth N. Wells
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include "Constant.h"
#include "Form.h"
#include "FunctionSpace.h"
#include "utils.h"
#include <dolfinx/common/IndexMap.h>
#include <dolfinx/common/utils.h>
#include <dolfinx/mesh/Geometry.h>
#include <dolfinx/mesh/Mesh.h>
#include <dolfinx/mesh/Topology.h>
#include <memory>
#include <vector>

namespace dolfinx::fem::impl
{

/// Assemble functional over cells
template <typename T>
T assemble_cells(const mesh::Geometry& geometry,
                 const xtl::span<const std::int32_t>& cells,
                 const std::function<void(T*, const T*, const T*, const double*,
                                          const int*, const std::uint8_t*)>& fn,
                 const xtl::span<const T>& constants,
                 const xtl::span<const T>& coeffs, int cstride)
{
  T value(0);
  if (cells.empty())
    return value;

  // Prepare cell geometry
  const graph::AdjacencyList<std::int32_t>& x_dofmap = geometry.dofmap();
  const std::size_t num_dofs_g = geometry.cmap().dim();
  xtl::span<const double> x_g = geometry.x();

  // Create data structures used in assembly
  std::vector<double> coordinate_dofs(3 * num_dofs_g);

  // Iterate over all cells
  for (std::size_t index = 0; index < cells.size(); ++index)
  {
    std::int32_t c = cells[index];

    // Get cell coordinates/geometry
    auto x_dofs = x_dofmap.links(c);
    for (std::size_t i = 0; i < x_dofs.size(); ++i)
    {
      common::impl::copy_N<3>(std::next(x_g.begin(), 3 * x_dofs[i]),
                              std::next(coordinate_dofs.begin(), 3 * i));
    }

    const T* coeff_cell = coeffs.data() + index * cstride;
    fn(&value, coeff_cell, constants.data(), coordinate_dofs.data(), nullptr,
       nullptr);
  }

  return value;
}

/// Execute kernel over exterior facets and accumulate result
template <typename T>
T assemble_exterior_facets(
    const mesh::Mesh& mesh,
    const xtl::span<const std::pair<std::int32_t, int>>& facets,
    const std::function<void(T*, const T*, const T*, const double*, const int*,
                             const std::uint8_t*)>& fn,
    const xtl::span<const T>& constants, const xtl::span<const T>& coeffs,
    int cstride)
{
  T value(0);
  if (facets.empty())
    return value;

  // Prepare cell geometry
  const graph::AdjacencyList<std::int32_t>& x_dofmap = mesh.geometry().dofmap();
  const std::size_t num_dofs_g = mesh.geometry().cmap().dim();
  xtl::span<const double> x_g = mesh.geometry().x();

  // Create data structures used in assembly
  std::vector<double> coordinate_dofs(3 * num_dofs_g);

  // Iterate over all facets
  for (std::size_t index = 0; index < facets.size(); ++index)
  {
    std::int32_t cell = facets[index].first;
    int local_facet = facets[index].second;

    // Get cell coordinates/geometry
    auto x_dofs = x_dofmap.links(cell);
    for (std::size_t i = 0; i < x_dofs.size(); ++i)
    {
      common::impl::copy_N<3>(std::next(x_g.begin(), 3 * x_dofs[i]),
                              std::next(coordinate_dofs.begin(), 3 * i));
    }

    const T* coeff_cell = coeffs.data() + index * cstride;
    fn(&value, coeff_cell, constants.data(), coordinate_dofs.data(),
       &local_facet, nullptr);
  }

  return value;
}

/// Assemble functional over interior facets
template <typename T>
T assemble_interior_facets(
    const mesh::Mesh& mesh,
    const xtl::span<const std::tuple<std::int32_t, int, std::int32_t, int>>&
        facets,
    const std::function<void(T*, const T*, const T*, const double*, const int*,
                             const std::uint8_t*)>& fn,
    const xtl::span<const T>& constants, const xtl::span<const T>& coeffs,
    int cstride, const xtl::span<const int>& offsets,
    const xtl::span<const std::uint8_t>& perms)
{
  T value(0);
  if (facets.empty())
    return value;

  const int tdim = mesh.topology().dim();

  // Prepare cell geometry
  const graph::AdjacencyList<std::int32_t>& x_dofmap = mesh.geometry().dofmap();
  const std::size_t num_dofs_g = mesh.geometry().cmap().dim();
  xtl::span<const double> x_g = mesh.geometry().x();

  // Create data structures used in assembly
  xt::xtensor<double, 3> coordinate_dofs({2, num_dofs_g, 3});
  std::vector<T> coeff_array(2 * offsets.back());
  assert(offsets.back() == cstride);

  const int num_cell_facets
      = mesh::cell_num_entities(mesh.topology().cell_type(), tdim - 1);

  // Iterate over all facets
  for (std::size_t index = 0; index < facets.size(); ++index)
  {
    const std::array<std::int32_t, 2> cells
        = {std::get<0>(facets[index]), std::get<2>(facets[index])};
    const std::array<int, 2> local_facet
        = {std::get<1>(facets[index]), std::get<3>(facets[index])};

    // Get cell geometry
    auto x_dofs0 = x_dofmap.links(cells[0]);
    for (std::size_t i = 0; i < x_dofs0.size(); ++i)
    {
      common::impl::copy_N<3>(
          std::next(x_g.begin(), 3 * x_dofs0[i]),
          xt::view(coordinate_dofs, 0, i, xt::all()).begin());
    }
    auto x_dofs1 = x_dofmap.links(cells[1]);
    for (std::size_t i = 0; i < x_dofs1.size(); ++i)
    {
      common::impl::copy_N<3>(
          std::next(x_g.begin(), 3 * x_dofs1[i]),
          xt::view(coordinate_dofs, 1, i, xt::all()).begin());
    }

    const std::array perm{perms[cells[0] * num_cell_facets + local_facet[0]],
                          perms[cells[1] * num_cell_facets + local_facet[1]]};
    fn(&value, coeffs.data() + index * 2 * cstride, constants.data(),
       coordinate_dofs.data(), local_facet.data(), perm.data());
  }

  return value;
}

/// Assemble functional into an scalar
template <typename T>
T assemble_scalar(
    const fem::Form<T>& M, const xtl::span<const T>& constants,
    const std::map<std::pair<IntegralType, int>,
                   std::pair<xtl::span<const T>, int>>& coefficients)
{
  std::shared_ptr<const mesh::Mesh> mesh = M.mesh();
  assert(mesh);

  T value = 0;
  for (int i : M.integral_ids(IntegralType::cell))
  {
    const auto& fn = M.kernel(IntegralType::cell, i);
    const auto& [coeffs, cstride] = coefficients.at({IntegralType::cell, i});
    const std::vector<std::int32_t>& cells = M.cell_domains(i);
    value += impl::assemble_cells(mesh->geometry(), cells, fn, constants,
                                  coeffs, cstride);
  }

  for (int i : M.integral_ids(IntegralType::exterior_facet))
  {
    const auto& fn = M.kernel(IntegralType::exterior_facet, i);
    const auto& [coeffs, cstride]
        = coefficients.at({IntegralType::exterior_facet, i});
    const std::vector<std::pair<std::int32_t, int>>& facets
        = M.exterior_facet_domains(i);
    value += impl::assemble_exterior_facets(*mesh, facets, fn, constants,
                                            coeffs, cstride);
  }

  if (M.num_integrals(IntegralType::interior_facet) > 0)
  {
    mesh->topology_mutable().create_entity_permutations();

    const std::vector<std::uint8_t>& perms
        = mesh->topology().get_facet_permutations();

    const std::vector<int> c_offsets = M.coefficient_offsets();
    for (int i : M.integral_ids(IntegralType::interior_facet))
    {
      const auto& fn = M.kernel(IntegralType::interior_facet, i);
      const auto& [coeffs, cstride]
          = coefficients.at({IntegralType::interior_facet, i});
      const std::vector<std::tuple<std::int32_t, int, std::int32_t, int>>&
          facets
          = M.interior_facet_domains(i);
      value += impl::assemble_interior_facets(
          *mesh, facets, fn, constants, coeffs, cstride, c_offsets, perms);
    }
  }

  return value;
}

} // namespace dolfinx::fem::impl
