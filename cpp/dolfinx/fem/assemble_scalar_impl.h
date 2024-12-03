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
#include <dolfinx/common/IndexMap.h>
#include <dolfinx/mesh/Geometry.h>
#include <dolfinx/mesh/Mesh.h>
#include <dolfinx/mesh/Topology.h>
#include <memory>
#include <vector>

namespace dolfinx::fem::impl
{

template <dolfinx::scalar T>
inline void get_cell_geometry(std::span<T> coordinate_dofs,
                              mdspan2_t x_dofmap,
                              std::span<const T> x,
                              std::int32_t cell) {
  auto x_dofs = MDSPAN_IMPL_STANDARD_NAMESPACE::submdspan(
      x_dofmap, cell, MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent);
  for (std::size_t i = 0; i < x_dofs.size(); ++i)
  {
    std::copy_n(std::next(x.begin(), 3 * x_dofs[i]), 3,
                std::next(coordinate_dofs.begin(), 3 * i));
  }
}

inline std::uint8_t get_cell_permutations(std::int32_t cell,
                                          std::int32_t local_facet,
                                          int num_facets_per_cell,
                                          std::span<const std::uint8_t> perms)
{
  return perms.empty() ? 0 : perms[cell * num_facets_per_cell + local_facet];
}

/// Assemble functional over cells
template <dolfinx::scalar T>
T assemble_cells(mdspan2_t x_dofmap, std::span<const scalar_value_type_t<T>> x,
                 std::span<const std::int32_t> cells, FEkernel<T> auto fn,
                 std::span<const T> constants, std::span<const T> coeffs,
                 int cstride)
{
  T value(0);
  if (cells.empty())
    return value;

  // Create data structures used in assembly
  std::vector<scalar_value_type_t<T>> coordinate_dofs(3 * x_dofmap.extent(1));

  // Iterate over all cells
  for (std::size_t index = 0; index < cells.size(); ++index)
  {
    std::int32_t cell = cells[index];

    // Get cell coordinates/geometry
    get_cell_geometry(std::span(coordinate_dofs), x_dofmap, x, cell);

    const T* coeff_cell = coeffs.data() + index * cstride;
    fn(&value, coeff_cell, constants.data(), coordinate_dofs.data(),
       nullptr, nullptr);
  }

  return value;
}

/// Execute kernel over exterior facets and accumulate result
template <dolfinx::scalar T>
T assemble_exterior_facets(mdspan2_t x_dofmap,
                           std::span<const scalar_value_type_t<T>> x,
                           int num_facets_per_cell,
                           std::span<const std::int32_t> facets,
                           FEkernel<T> auto fn, std::span<const T> constants,
                           std::span<const T> coeffs, int cstride,
                           std::span<const std::uint8_t> perms)
{
  T value(0);
  if (facets.empty())
    return value;

  // Create data structures used in assembly
  std::vector<scalar_value_type_t<T>> coordinate_dofs(3 * x_dofmap.extent(1));

  // Iterate over all facets
  const std::uint8_t step = 4;
  assert(facets.size() % step == 0);
  for (std::size_t index = 0; index < facets.size(); index += step)
  {
    std::int32_t cell = facets[index];
    std::int32_t local_facet = facets[index + 1];

    // Get cell coordinates/geometry
    get_cell_geometry(std::span(coordinate_dofs), x_dofmap, x, cell);

    // Permutations
    auto perm = get_cell_permutations(cell, local_facet, num_facets_per_cell, perms);

    const T* coeff_cell = coeffs.data() + index / 2 * cstride;
    fn(&value, coeff_cell, constants.data(), coordinate_dofs.data(),
       &local_facet, &perm);
  }

  return value;
}

/// Assemble functional over interior facets
template <dolfinx::scalar T>
T assemble_interior_facets(mdspan2_t x_dofmap,
                           std::span<const scalar_value_type_t<T>> x,
                           int num_facets_per_cell,
                           std::span<const std::int32_t> facets,
                           FEkernel<T> auto fn, std::span<const T> constants,
                           std::span<const T> coeffs, int cstride,
                           std::span<const int> offsets,
                           std::span<const std::uint8_t> perms)
{
  T value(0);
  if (facets.empty())
    return value;

  // Create data structures used in assembly
  auto dofs_size = 3 * x_dofmap.extent(1);
  using X = scalar_value_type_t<T>;
  std::vector<X> coordinate_dofs_cnt(2 * dofs_size);
  std::span<X> coordinate_dofs(coordinate_dofs_cnt);

  std::vector<T> coeff_array(2 * offsets.back());
  assert(offsets.back() == cstride);

  // Iterate over all facets
  const std::uint8_t step = 4;
  assert(facets.size() % step == 0);
  for (std::size_t index = 0; index < facets.size(); index += step)
  {
    std::array<std::int32_t, 2> cells = {facets[index], facets[index + 2]};
    std::array<std::int32_t, 2> local_facets = {facets[index + 1], facets[index + 3]};

    // Get cell coordinates/geometry
    get_cell_geometry(coordinate_dofs.first(dofs_size), x_dofmap, x, cells[0]);
    get_cell_geometry(coordinate_dofs.last(dofs_size), x_dofmap, x, cells[1]);

    // Permutations
    auto perm = std::array<std::uint8_t, 2>{
      get_cell_permutations(cells[0], local_facets[0], num_facets_per_cell, perms),
      get_cell_permutations(cells[1], local_facets[1], num_facets_per_cell, perms)
    };

    const T* coeff_cell = coeffs.data() + index / 2 * cstride;
    fn(&value, coeff_cell, constants.data(), coordinate_dofs.data(),
      local_facets.data(), perm.data());
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
    auto fn = M.kernel(IntegralType::cell, i);
    assert(fn);
    auto& [coeffs, cstride] = coefficients.at({IntegralType::cell, i});
    std::span<const std::int32_t> cells = M.domain(IntegralType::cell, i);
    value += impl::assemble_cells(x_dofmap, x, cells, fn, constants, coeffs,
                                  cstride);
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
    auto fn = M.kernel(IntegralType::exterior_facet, i);
    assert(fn);
    auto& [coeffs, cstride] = coefficients.at({IntegralType::exterior_facet, i});
    value += impl::assemble_exterior_facets(
        x_dofmap, x, num_facets_per_cell,
        M.domain(IntegralType::exterior_facet, i), fn, constants, coeffs,
        cstride, perms);
  }

  for (int i : M.integral_ids(IntegralType::interior_facet))
  {
    const std::vector<int> c_offsets = M.coefficient_offsets();
    auto fn = M.kernel(IntegralType::interior_facet, i);
    assert(fn);
    auto& [coeffs, cstride] = coefficients.at({IntegralType::interior_facet, i});
    value += impl::assemble_interior_facets(
        x_dofmap, x, num_facets_per_cell,
        M.domain(IntegralType::interior_facet, i), fn, constants, coeffs,
        cstride, c_offsets, perms);
  }

  return value;
}

} // namespace dolfinx::fem::impl
