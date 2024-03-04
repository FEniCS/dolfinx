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
    std::int32_t c = cells[index];

    // Get cell coordinates/geometry
    auto x_dofs = MDSPAN_IMPL_STANDARD_NAMESPACE::submdspan(
        x_dofmap, c, MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent);
    for (std::size_t i = 0; i < x_dofs.size(); ++i)
    {
      std::copy_n(std::next(x.begin(), 3 * x_dofs[i]), 3,
                  std::next(coordinate_dofs.begin(), 3 * i));
    }

    const T* coeff_cell = coeffs.data() + index * cstride;
    fn(&value, coeff_cell, constants.data(), coordinate_dofs.data(), nullptr,
       nullptr);
  }

  return value;
}

/// Execute kernel over exterior facets and accumulate result
template <dolfinx::scalar T>
T assemble_exterior_facets(mdspan2_t x_dofmap,
                           std::span<const scalar_value_type_t<T>> x,
                           std::span<const std::int32_t> facets,
                           FEkernel<T> auto fn, std::span<const T> constants,
                           std::span<const T> coeffs, int cstride)
{
  T value(0);
  if (facets.empty())
    return value;

  // Create data structures used in assembly
  std::vector<scalar_value_type_t<T>> coordinate_dofs(3 * x_dofmap.extent(1));

  // Iterate over all facets
  assert(facets.size() % 2 == 0);
  for (std::size_t index = 0; index < facets.size(); index += 2)
  {
    std::int32_t cell = facets[index];
    std::int32_t local_facet = facets[index + 1];

    // Get cell coordinates/geometry
    auto x_dofs = MDSPAN_IMPL_STANDARD_NAMESPACE::submdspan(
        x_dofmap, cell, MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent);
    for (std::size_t i = 0; i < x_dofs.size(); ++i)
    {
      std::copy_n(std::next(x.begin(), 3 * x_dofs[i]), 3,
                  std::next(coordinate_dofs.begin(), 3 * i));
    }

    const T* coeff_cell = coeffs.data() + index / 2 * cstride;
    fn(&value, coeff_cell, constants.data(), coordinate_dofs.data(),
       &local_facet, nullptr);
  }

  return value;
}

/// Assemble functional over interior facets
template <dolfinx::scalar T>
T assemble_interior_facets(mdspan2_t x_dofmap,
                           std::span<const scalar_value_type_t<T>> x,
                           int num_cell_facets,
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
  using X = scalar_value_type_t<T>;
  std::vector<X> coordinate_dofs(2 * x_dofmap.extent(1) * 3);
  std::span<X> cdofs0(coordinate_dofs.data(), x_dofmap.extent(1) * 3);
  std::span<X> cdofs1(coordinate_dofs.data() + x_dofmap.extent(1) * 3,
                      x_dofmap.extent(1) * 3);

  std::vector<T> coeff_array(2 * offsets.back());
  assert(offsets.back() == cstride);

  // Iterate over all facets
  assert(facets.size() % 4 == 0);
  for (std::size_t index = 0; index < facets.size(); index += 4)
  {
    std::array<std::int32_t, 2> cells = {facets[index], facets[index + 2]};
    std::array<std::int32_t, 2> local_facet
        = {facets[index + 1], facets[index + 3]};

    // Get cell geometry
    auto x_dofs0 = MDSPAN_IMPL_STANDARD_NAMESPACE::submdspan(
        x_dofmap, cells[0], MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent);
    for (std::size_t i = 0; i < x_dofs0.size(); ++i)
    {
      std::copy_n(std::next(x.begin(), 3 * x_dofs0[i]), 3,
                  std::next(cdofs0.begin(), 3 * i));
    }
    auto x_dofs1 = MDSPAN_IMPL_STANDARD_NAMESPACE::submdspan(
        x_dofmap, cells[1], MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent);
    for (std::size_t i = 0; i < x_dofs1.size(); ++i)
    {
      std::copy_n(std::next(x.begin(), 3 * x_dofs1[i]), 3,
                  std::next(cdofs1.begin(), 3 * i));
    }

    const std::array perm{perms[cells[0] * num_cell_facets + local_facet[0]],
                          perms[cells[1] * num_cell_facets + local_facet[1]]};
    fn(&value, coeffs.data() + index / 2 * cstride, constants.data(),
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
    auto fn = M.kernel(IntegralType::cell, i);
    assert(fn);
    auto& [coeffs, cstride] = coefficients.at({IntegralType::cell, i});
    std::span<const std::int32_t> cells = M.domain(IntegralType::cell, i);
    value += impl::assemble_cells(x_dofmap, x, cells, fn, constants, coeffs,
                                  cstride);
  }

  for (int i : M.integral_ids(IntegralType::exterior_facet))
  {
    auto fn = M.kernel(IntegralType::exterior_facet, i);
    assert(fn);
    auto& [coeffs, cstride]
        = coefficients.at({IntegralType::exterior_facet, i});
    value += impl::assemble_exterior_facets(
        x_dofmap, x, M.domain(IntegralType::exterior_facet, i), fn, constants,
        coeffs, cstride);
  }

  if (M.num_integrals(IntegralType::interior_facet) > 0)
  {
    mesh->topology_mutable()->create_entity_permutations();
    const std::vector<std::uint8_t>& perms
        = mesh->topology()->get_facet_permutations();
    mesh::CellType cell_type = mesh->topology()->cell_type();
    int num_cell_facets
        = mesh::cell_num_entities(cell_type, mesh->topology()->dim() - 1);
    const std::vector<int> c_offsets = M.coefficient_offsets();
    for (int i : M.integral_ids(IntegralType::interior_facet))
    {
      auto fn = M.kernel(IntegralType::interior_facet, i);
      assert(fn);
      auto& [coeffs, cstride]
          = coefficients.at({IntegralType::interior_facet, i});
      value += impl::assemble_interior_facets(
          x_dofmap, x, num_cell_facets,
          M.domain(IntegralType::interior_facet, i), fn, constants, coeffs,
          cstride, c_offsets, perms);
    }
  }

  return value;
}

} // namespace dolfinx::fem::impl
