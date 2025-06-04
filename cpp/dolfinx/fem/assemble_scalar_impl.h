// Copyright (C) 2019-2025 Garth N. Wells
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
/// @brief Assemble functional over cells.
/// @tparam T Scalar type.
/// @param[in] x_dofmap Dofmap for the mesh geometry.
/// @param[in] x Mesh geometry (coordinates).
/// @param[in] cells Cell indices to execute the kernel over. These are
/// the indices into the geometry dofmap `x_dofmap`.
/// @param[in] fn Kernel function to execute over each cell.
/// @param[in] constants Constant data in the kernel.
/// @param[in] coeffs Coefficient data in the kernel. It has shape
/// `(cells.size(), num_cell_coeffs)`. `coeffs(i, j)` is the `j`th
/// coefficient for cell `i`.
/// @param[in] cdofs Buffer array with size at least `3 *
/// x_dofmap.extent(1)`.
/// @return The contribution to the form (functional) from the cells
/// local to the process.
template <dolfinx::scalar T>
T assemble_cells(mdspan2_t x_dofmap,
                 md::mdspan<const scalar_value_t<T>,
                            md::extents<std::size_t, md::dynamic_extent, 3>>
                     x,
                 std::span<const std::int32_t> cells, FEkernel<T> auto&& fn,
                 std::span<const T> constants,
                 md::mdspan<const T, md::dextents<std::size_t, 2>> coeffs,
                 std::span<scalar_value_t<T>> cdofs)
{
  T value(0);
  if (cells.empty())
    return value;

  assert(cdofs.sieze() >= 3 * x_dofmap.extent(1));

  // Iterate over all cells
  for (std::size_t index = 0; index < cells.size(); ++index)
  {
    std::int32_t c = cells[index];
    auto x_dofs = md::submdspan(x_dofmap, c, md::full_extent);
    for (std::size_t i = 0; i < x_dofs.size(); ++i)
      std::copy_n(&x(x_dofs[i], 0), 3, std::next(cdofs.begin(), 3 * i));
    fn(&value, &coeffs(index, 0), constants.data(), cdofs.data(), nullptr,
       nullptr, nullptr);
  }

  return value;
}

/// @brief Assemble functional over exterior facets.
/// @tparam T Scalar type.
/// @param[in] x_dofmap Dofmap for the mesh geometry.
/// @param[in] x Mesh geometry (coordinates).
/// @param[in] facets Facets (in the integration domain mesh) to execute
/// the kernel over.
/// @param[in] fn Kernel function to execute over each cell.
/// @param[in] constants Constant data in the kernel.
/// @param[in] coeffs The coefficient data array of shape
/// `(cells.size(), coeffs_per_cell)`.
/// @param[in] perms Facet permutation data. Empty if facet
/// permutations are not required.
/// @param[in] cdofs  Buffer array with size at least `3 *
/// x_dofmap.extent(1)`.
/// @return The contribution to the form (functional) from the exterior
/// cells local to the process.
template <dolfinx::scalar T>
T assemble_exterior_facets(
    mdspan2_t x_dofmap,
    md::mdspan<const scalar_value_t<T>,
               md::extents<std::size_t, md::dynamic_extent, 3>>
        x,
    md::mdspan<const std::int32_t,
               md::extents<std::size_t, md::dynamic_extent, 2>>
        facets,
    FEkernel<T> auto fn, std::span<const T> constants,
    md::mdspan<const T, md::dextents<std::size_t, 2>> coeffs,
    md::mdspan<const std::uint8_t, md::dextents<std::size_t, 2>> perms,
    std::span<scalar_value_t<T>> cdofs)
{
  T value(0);
  if (facets.empty())
    return value;

  assert(cdofs.sieze() >= 3 * x_dofmap.extent(1));

  // Iterate over all facets
  for (std::size_t f = 0; f < facets.extent(0); ++f)
  {
    std::int32_t cell = facets(f, 0);
    std::int32_t local_facet = facets(f, 1);

    // Get cell coordinates/geometry
    auto x_dofs = md::submdspan(x_dofmap, cell, md::full_extent);
    for (std::size_t i = 0; i < x_dofs.size(); ++i)
      std::copy_n(&x(x_dofs[i], 0), 3, std::next(cdofs.begin(), 3 * i));

    // Permutations
    std::uint8_t perm = perms.empty() ? 0 : perms(cell, local_facet);
    fn(&value, &coeffs(f, 0), constants.data(), cdofs.data(), &local_facet,
       &perm, nullptr);
  }

  return value;
}

/// @brief Assemble functional over interior facets.
/// @tparam T Scalar type.
/// @param[in] x_dofmap Dofmap for the mesh geometry.
/// @param[in] x  Mesh geometry (coordinates).
/// @param[in] facets Facets (in the integration domain mesh) to execute
/// the kernel over.
/// @param[in] fn Kernel function to execute over each facet.
/// @param[in] constants Constant data in the kernel.
/// @param[in] coeffsCoefficient data array of shape `(cells.size(),
/// coeffs_per_cell)`.
/// @param[in] perms Facet permutation data. Empty if facet
/// permutations are not required.
/// @param[in] cdofs Buffer array with size at least `6 *
/// x_dofmap.extent(1)`.
/// @return The contribution to the form (functional) from the interior
/// cells local to the process.
template <dolfinx::scalar T>
T assemble_interior_facets(
    mdspan2_t x_dofmap,
    md::mdspan<const scalar_value_t<T>,
               md::extents<std::size_t, md::dynamic_extent, 3>>
        x,
    md::mdspan<const std::int32_t,
               md::extents<std::size_t, md::dynamic_extent, 2, 2>>
        facets,
    FEkernel<T> auto&& fn, std::span<const T> constants,
    md::mdspan<const T, md::extents<std::size_t, md::dynamic_extent, 2,
                                    md::dynamic_extent>>
        coeffs,
    md::mdspan<const std::uint8_t, md::dextents<std::size_t, 2>> perms,
    std::span<scalar_value_t<T>> cdofs)
{
  T value(0);
  if (facets.empty())
    return value;

  // Create data structures used in assembly
  assert(cdofs.size() >= 2 * 3 * x_dofmap.extent(1));
  auto cdofs0 = cdofs.subspan(0, 3 * x_dofmap.extent(1));
  auto cdofs1 = cdofs.subspan(3 * x_dofmap.extent(1), 3 * x_dofmap.extent(1));

  // Iterate over all facets
  for (std::size_t f = 0; f < facets.extent(0); ++f)
  {
    std::array cells = {facets(f, 0, 0), facets(f, 1, 0)};
    std::array local_facet = {facets(f, 0, 1), facets(f, 1, 1)};

    // Get cell geometry
    auto x_dofs0 = md::submdspan(x_dofmap, cells[0], md::full_extent);
    for (std::size_t i = 0; i < x_dofs0.size(); ++i)
      std::copy_n(&x(x_dofs0[i], 0), 3, std::next(cdofs0.begin(), 3 * i));
    auto x_dofs1 = md::submdspan(x_dofmap, cells[1], md::full_extent);
    for (std::size_t i = 0; i < x_dofs1.size(); ++i)
      std::copy_n(&x(x_dofs1[i], 0), 3, std::next(cdofs1.begin(), 3 * i));

    std::array perm = perms.empty()
                          ? std::array<std::uint8_t, 2>{0, 0}
                          : std::array{perms(cells[0], local_facet[0]),
                                       perms(cells[1], local_facet[1])};
    fn(&value, &coeffs(f, 0, 0), constants.data(), cdofs.data(),
       local_facet.data(), perm.data(), nullptr);
  }

  return value;
}

/// Assemble functional into an scalar with provided mesh geometry.
template <dolfinx::scalar T, std::floating_point U>
T assemble_scalar(
    const fem::Form<T, U>& M, mdspan2_t x_dofmap,
    md::mdspan<const scalar_value_t<T>,
               md::extents<std::size_t, md::dynamic_extent, 3>>
        x,
    std::span<const T> constants,
    const std::map<std::pair<IntegralType, int>,
                   std::pair<std::span<const T>, int>>& coefficients)
{
  std::shared_ptr<const mesh::Mesh<U>> mesh = M.mesh();
  assert(mesh);

  md::mdspan<const std::uint8_t, md::dextents<std::size_t, 2>> perms;
  if (M.needs_facet_permutations())
  {
    mesh::CellType cell_type = mesh->topology()->cell_type();
    int num_facets_per_cell
        = mesh::cell_num_entities(cell_type, mesh->topology()->dim() - 1);
    mesh->topology_mutable()->create_entity_permutations();
    const std::vector<std::uint8_t>& p
        = mesh->topology()->get_facet_permutations();
    perms = md::mdspan(p.data(), p.size() / num_facets_per_cell,
                       num_facets_per_cell);
  }

  T value = 0;

  {
    std::vector<scalar_value_t<T>> cdofs_b(3 * x_dofmap.extent(1));

    for (int i : M.integral_ids(IntegralType::cell))
    {
      auto fn = M.kernel(IntegralType::cell, i, 0);
      assert(fn);
      auto& [coeffs, cstride] = coefficients.at({IntegralType::cell, i});
      std::span<const std::int32_t> cells = M.domain(IntegralType::cell, i, 0);
      assert(cells.size() * cstride == coeffs.size());
      value += impl::assemble_cells(
          x_dofmap, x, cells, fn, constants,
          md::mdspan(coeffs.data(), cells.size(), cstride), cdofs_b);
    }

    for (int i : M.integral_ids(IntegralType::exterior_facet))
    {
      auto fn = M.kernel(IntegralType::exterior_facet, i, 0);
      assert(fn);
      auto& [coeffs, cstride]
          = coefficients.at({IntegralType::exterior_facet, i});

      std::span facets = M.domain(IntegralType::exterior_facet, i, 0);
      assert((facets.size() / 2) * cstride == coeffs.size());
      value += impl::assemble_exterior_facets(
          x_dofmap, x,
          md::mdspan<const std::int32_t,
                     md::extents<std::size_t, md::dynamic_extent, 2>>(
              facets.data(), facets.size() / 2, 2),
          fn, constants, md::mdspan(coeffs.data(), facets.size() / 2, cstride),
          perms, cdofs_b);
    }
  }

  {
    std::vector<scalar_value_t<T>> cdofs_b(2 * 3 * x_dofmap.extent(1));

    for (int i : M.integral_ids(IntegralType::interior_facet))
    {
      auto fn = M.kernel(IntegralType::interior_facet, i, 0);
      assert(fn);
      auto& [coeffs, cstride]
          = coefficients.at({IntegralType::interior_facet, i});
      std::span facets = M.domain(IntegralType::interior_facet, i, 0);
      assert((facets.size() / 4) * 2 * cstride == coeffs.size());
      value += impl::assemble_interior_facets(
          x_dofmap, x,
          md::mdspan<const std::int32_t,
                     md::extents<std::size_t, md::dynamic_extent, 2, 2>>(
              facets.data(), facets.size() / 4, 2, 2),
          fn, constants,
          md::mdspan<const T, md::extents<std::size_t, md::dynamic_extent, 2,
                                          md::dynamic_extent>>(
              coeffs.data(), facets.size() / 4, 2, cstride),
          perms, cdofs_b);
    }
  }

  return value;
}

} // namespace dolfinx::fem::impl
