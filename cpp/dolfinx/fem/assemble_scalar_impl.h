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
#include <algorithm>
#include <basix/mdspan.hpp>
#include <dolfinx/common/IndexMap.h>
#include <dolfinx/mesh/Geometry.h>
#include <dolfinx/mesh/Mesh.h>
#include <dolfinx/mesh/Topology.h>
#include <memory>
#include <vector>

namespace dolfinx::fem::impl
{
namespace md = MDSPAN_IMPL_STANDARD_NAMESPACE;

/// Assemble functional over cells
template <dolfinx::scalar T>
T assemble_cells(mdspan2_t x_dofmap, std::span<const scalar_value_type_t<T>> x,
                 std::span<const std::int32_t> cells, FEkernel<T> auto fn,
                 std::span<const T> constants,
                 md::mdspan<const T, md::dextents<std::size_t, 2>> coeffs)
{

  T value(0);
  if (cells.empty())
    return value;

  // Create data structures used in assembly
  std::vector<scalar_value_type_t<T>> coordinate_dofs(3 * x_dofmap.extent(1));

  // Iterate over all cells
  for (std::size_t index = 0; index < cells.size(); ++index)
  {
    std::int32_t c = cells[index];

    // Get cell coordinates/geometry
    auto x_dofs = md::submdspan(x_dofmap, c, md::full_extent);
    for (std::size_t i = 0; i < x_dofs.size(); ++i)
    {
      std::copy_n(std::next(x.begin(), 3 * x_dofs[i]), 3,
                  std::next(coordinate_dofs.begin(), 3 * i));
    }

    fn(&value, &coeffs(index, 0), constants.data(), coordinate_dofs.data(),
       nullptr, nullptr);
  }

  return value;
}

/// Execute kernel over exterior facets and accumulate result
template <dolfinx::scalar T>
T assemble_exterior_facets(
    mdspan2_t x_dofmap, std::span<const scalar_value_type_t<T>> x,
    int num_facets_per_cell,
    md::mdspan<const std::int32_t,
               md::extents<std::size_t, md::dynamic_extent, 2>>
        facets,
    FEkernel<T> auto fn, std::span<const T> constants,
    md::mdspan<const T, md::dextents<std::size_t, 2>> coeffs,
    std::span<const std::uint8_t> perms)
{
  T value(0);
  if (facets.empty())
    return value;

  // Create data structures used in assembly
  std::vector<scalar_value_type_t<T>> coordinate_dofs(3 * x_dofmap.extent(1));

  // Iterate over all facets
  for (std::size_t f = 0; f < facets.extent(0); ++f)
  {
    std::int32_t cell = facets(f, 0);
    std::int32_t local_facet = facets(f, 1);

    // Get cell coordinates/geometry
    auto x_dofs = md::submdspan(x_dofmap, cell, md::full_extent);
    for (std::size_t i = 0; i < x_dofs.size(); ++i)
    {
      std::copy_n(std::next(x.begin(), 3 * x_dofs[i]), 3,
                  std::next(coordinate_dofs.begin(), 3 * i));
    }

    // Permutations
    std::uint8_t perm
        = perms.empty() ? 0 : perms[cell * num_facets_per_cell + local_facet];
    fn(&value, &coeffs(f, 0), constants.data(), coordinate_dofs.data(),
       &local_facet, &perm);
  }

  return value;
}

/// Assemble functional over interior facets
template <dolfinx::scalar T>
T assemble_interior_facets(
    mdspan2_t x_dofmap, std::span<const scalar_value_type_t<T>> x,
    int num_facets_per_cell,
    md::mdspan<const std::int32_t,
               md::extents<std::size_t, md::dynamic_extent, 4>>
        facets,
    FEkernel<T> auto fn, std::span<const T> constants,
    std::span<const T> coeffs, int cstride, std::span<const int> offsets,
    std::span<const std::uint8_t> perms)
{
  T value(0);
  if (facets.empty())
    return value;

  assert(offsets.back() == cstride);

  // Create data structures used in assembly
  using X = scalar_value_type_t<T>;
  std::vector<X> coordinate_dofs(2 * x_dofmap.extent(1) * 3);
  std::span<X> cdofs0(coordinate_dofs.data(), x_dofmap.extent(1) * 3);
  std::span<X> cdofs1(coordinate_dofs.data() + x_dofmap.extent(1) * 3,
                      x_dofmap.extent(1) * 3);

  // Iterate over all facets
  for (std::size_t f = 0; f < facets.extent(0); ++f)
  {
    std::array cells = {facets(f, 0), facets(f, 2)};
    std::array local_facet = {facets(f, 1), facets(f, 3)};

    // Get cell geometry
    auto x_dofs0 = md::submdspan(x_dofmap, cells[0], md::full_extent);
    for (std::size_t i = 0; i < x_dofs0.size(); ++i)
    {
      std::copy_n(std::next(x.begin(), 3 * x_dofs0[i]), 3,
                  std::next(cdofs0.begin(), 3 * i));
    }
    auto x_dofs1 = md::submdspan(x_dofmap, cells[1], md::full_extent);
    for (std::size_t i = 0; i < x_dofs1.size(); ++i)
    {
      std::copy_n(std::next(x.begin(), 3 * x_dofs1[i]), 3,
                  std::next(cdofs1.begin(), 3 * i));
    }

    std::array perm
        = perms.empty()
              ? std::array<std::uint8_t, 2>{0, 0}
              : std::array{
                    perms[cells[0] * num_facets_per_cell + local_facet[0]],
                    perms[cells[1] * num_facets_per_cell + local_facet[1]]};
    fn(&value, coeffs.data() + 2 * f * cstride, constants.data(),
       coordinate_dofs.data(), local_facet.data(), perm.data());
  }

  return value;
}

/// Assemble functional into an scalar with provided mesh geometry.
template <dolfinx::scalar T, std::floating_point U>
T assemble_scalar(
    const fem::Form<T, U>& M, mdspan2_t x_dofmap,
    std::span<const scalar_value_type_t<T>> x, std::span<const T> constants,
    const std::map<std::pair<IntegralType, int>,
                   std::pair<std::span<const T>, int>>& coefficients)
{
  std::shared_ptr<const mesh::Mesh<U>> mesh = M.mesh();
  assert(mesh);

  T value = 0;
  for (int i : M.integral_ids(IntegralType::cell))
  {
    auto fn = M.kernel(IntegralType::cell, i, 0);
    assert(fn);
    auto& [coeffs, cstride] = coefficients.at({IntegralType::cell, i});
    std::span<const std::int32_t> cells = M.domain(IntegralType::cell, i, 0);
    assert(cells.size() * cstride == coeffs.size());
    value += impl::assemble_cells(
        x_dofmap, x, cells, fn, constants,
        md::mdspan(coeffs.data(), cells.size(), cstride));
  }

  std::span<const std::uint8_t> perms;
  if (M.needs_facet_permutations())
  {
    mesh->topology_mutable()->create_entity_permutations();
    perms = std::span(mesh->topology()->get_facet_permutations());
  }

  mesh::CellType cell_type = mesh->topology()->cell_type();
  int num_facets_per_cell
      = mesh::cell_num_entities(cell_type, mesh->topology()->dim() - 1);
  for (int i : M.integral_ids(IntegralType::exterior_facet))
  {
    auto fn = M.kernel(IntegralType::exterior_facet, i, 0);
    assert(fn);
    auto& [coeffs, cstride]
        = coefficients.at({IntegralType::exterior_facet, i});

    std::span facets = M.domain(IntegralType::exterior_facet, i, 0);
    assert((facets.size() / 2) * cstride == coeffs.size());
    value += impl::assemble_exterior_facets(
        x_dofmap, x, num_facets_per_cell,
        md::mdspan<const std::int32_t,
                   md::extents<std::size_t, md::dynamic_extent, 2>>(
            facets.data(), facets.size() / 2, 2),
        fn, constants, md::mdspan(coeffs.data(), facets.size() / 2, cstride),
        perms);
  }

  for (int i : M.integral_ids(IntegralType::interior_facet))
  {
    const std::vector<int> c_offsets = M.coefficient_offsets();
    auto fn = M.kernel(IntegralType::interior_facet, i, 0);
    assert(fn);
    auto& [coeffs, cstride]
        = coefficients.at({IntegralType::interior_facet, i});

    std::span facets = M.domain(IntegralType::interior_facet, i, 0);

    value += impl::assemble_interior_facets(
        x_dofmap, x, num_facets_per_cell,
        md::mdspan<const std::int32_t,
                   md::extents<std::size_t, md::dynamic_extent, 4>>(
            facets.data(), facets.size() / 4, 4),
        fn, constants, coeffs, cstride, c_offsets, perms);
  }

  return value;
}

} // namespace dolfinx::fem::impl
