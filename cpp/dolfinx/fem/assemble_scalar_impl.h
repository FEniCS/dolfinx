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
#include <variant>
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

template <dolfinx::scalar T>
inline T* get_value_ptr(std::variant<std::shared_ptr<T>, std::span<T>> acc, std::size_t i)
{
  if (std::holds_alternative<std::shared_ptr<T>>(acc))
  {
    return std::get<std::shared_ptr<T>>(acc).get();
  }
  else if (std::holds_alternative<std::span<T>>(acc)){
    return &(std::get<std::span<T>>(acc)[i]);
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
void assemble_cells(std::variant<std::shared_ptr<T>, std::span<T>> acc,
                    mdspan2_t x_dofmap, std::span<const scalar_value_type_t<T>> x,
                    std::span<const std::int32_t> cells, FEkernel<T> auto fn,
                    std::span<const T> constants, std::span<const T> coeffs,
                    int cstride)
{
  if (cells.empty())
    return;

  if (!(std::holds_alternative<std::shared_ptr<T>>(acc) || std::holds_alternative<std::span<T>>(acc)))
    return;

  // Create data structures used in assembly
  std::vector<scalar_value_type_t<T>> coordinate_dofs(3 * x_dofmap.extent(1));

  // Iterate over all cells
  for (std::size_t index = 0; index < cells.size(); ++index)
  {
    std::int32_t cell = cells[index];

    // Get cell coordinates/geometry
    get_cell_geometry(std::span(coordinate_dofs), x_dofmap, x, cell);
    // Get cell form coefficients
    const T* coeff_cell = coeffs.data() + index * cstride;
    // Get output value pointer
    T* value = get_value_ptr(acc, index);

    fn(value, coeff_cell, constants.data(), coordinate_dofs.data(),
       nullptr, nullptr);
  }
}

/// Execute kernel over exterior facets and accumulate result
template <dolfinx::scalar T>
void assemble_exterior_facets(std::variant<std::shared_ptr<T>, std::span<T>> acc,
                              mdspan2_t x_dofmap,
                              std::span<const scalar_value_type_t<T>> x,
                              int num_facets_per_cell,
                              std::span<const std::int32_t> facets,
                              FEkernel<T> auto fn, std::span<const T> constants,
                              std::span<const T> coeffs, int cstride,
                              std::span<const std::uint8_t> perms)
{
  if (facets.empty())
    return;

  if (!(std::holds_alternative<std::shared_ptr<T>>(acc) || std::holds_alternative<std::span<T>>(acc)))
    return;

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
    // Get cell permutations
    std::uint8_t perm = get_cell_permutations(cell, local_facet, num_facets_per_cell, perms);
    // Get cell form coefficients
    const T* coeff_cell = coeffs.data() + index / 2 * cstride;
    // Get output value pointer
    T* value = get_value_ptr(acc, index);

    fn(value, coeff_cell, constants.data(), coordinate_dofs.data(),
       &local_facet, &perm);
  }
}

/// Assemble functional over interior facets
template <dolfinx::scalar T>
void assemble_interior_facets(std::variant<std::shared_ptr<T>, std::span<T>> acc,
                              mdspan2_t x_dofmap,
                              std::span<const scalar_value_type_t<T>> x,
                              int num_facets_per_cell,
                              std::span<const std::int32_t> facets,
                              FEkernel<T> auto fn, std::span<const T> constants,
                              std::span<const T> coeffs, int cstride,
                              std::span<const int> offsets,
                              std::span<const std::uint8_t> perms)
{
  if (facets.empty())
    return;

  if (!(std::holds_alternative<std::shared_ptr<T>>(acc) || std::holds_alternative<std::span<T>>(acc)))
    return;

  // Create data structures used in assembly
  const std::uint8_t dofs_size = 3 * x_dofmap.extent(1);
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
    // Get cell permutations
    std::array<std::uint8_t, 2> perm = {
      get_cell_permutations(cells[0], local_facets[0], num_facets_per_cell, perms),
      get_cell_permutations(cells[1], local_facets[1], num_facets_per_cell, perms)
    };
    // Get cell form coefficients
    const T* coeff_cell = coeffs.data() + index / 2 * cstride;
    // Get output value pointer
    T* value = get_value_ptr(acc, index);

    fn(value, coeff_cell, constants.data(), coordinate_dofs.data(), 
       local_facets.data(), perm.data());
  }
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

  std::shared_ptr<T> value_ptr = std::make_shared<T>(T(0));
  for (int i : M.integral_ids(IntegralType::cell))
  {
    auto fn = M.kernel(IntegralType::cell, i);
    assert(fn);
    auto& [coeffs, cstride] = coefficients.at({IntegralType::cell, i});
    std::span<const std::int32_t> cells = M.domain(IntegralType::cell, i);
    impl::assemble_cells<T>(value_ptr, x_dofmap, x, cells, fn, constants, coeffs, cstride);    
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
    std::span<const std::int32_t> exterior_facets = M.domain(IntegralType::exterior_facet, i);
    impl::assemble_exterior_facets<T>(value_ptr, x_dofmap, x, num_facets_per_cell, exterior_facets,
                                      fn, constants, coeffs, cstride, perms);
  }

  for (int i : M.integral_ids(IntegralType::interior_facet))
  {
    const std::vector<int> c_offsets = M.coefficient_offsets();
    auto fn = M.kernel(IntegralType::interior_facet, i);
    assert(fn);
    auto& [coeffs, cstride] = coefficients.at({IntegralType::interior_facet, i});
    std::span<const std::int32_t> interior_facets = M.domain(IntegralType::interior_facet, i);
    impl::assemble_interior_facets<T>(value_ptr, x_dofmap, x, num_facets_per_cell, interior_facets,
                                      fn, constants, coeffs, cstride, c_offsets, perms);
  }
  return *value_ptr;
}


/// Integrate functional over mesh geometry into a scalar vectors.
template <dolfinx::scalar T, std::floating_point U>
std::map<std::pair<IntegralType, int>, std::span<T>> integrate_scalar(
    const fem::Form<T, U>& M, mdspan2_t x_dofmap,
    std::span<const scalar_value_type_t<T>> x, std::span<const T> constants,
    const std::map<std::pair<IntegralType, int>,
                   std::pair<std::span<const T>, int>>& coefficients)
{
  std::shared_ptr<const mesh::Mesh<U>> mesh = M.mesh();
  assert(mesh);

  std::map<std::pair<IntegralType, int>, std::span<T>> values;
  
  for (int i : M.integral_ids(IntegralType::cell))
  {
    auto fn = M.kernel(IntegralType::cell, i);
    assert(fn);
    auto& [coeffs, cstride] = coefficients.at({IntegralType::cell, i});
    std::span<const std::int32_t> cells = M.domain(IntegralType::cell, i);

    std::vector<T> scalars(cells.size());
    impl::assemble_cells<T>(std::span<T>(scalars), x_dofmap, x, cells, fn, constants, coeffs, cstride);
    values.emplace(std::make_pair(IntegralType::cell, i), std::span<T>(scalars));
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
    std::span<const std::int32_t> exterior_facets = M.domain(IntegralType::exterior_facet, i);

    std::vector<T> scalars(exterior_facets.size());
    impl::assemble_exterior_facets<T>(std::span<T>(scalars), x_dofmap, x, num_facets_per_cell, exterior_facets,
                                      fn, constants, coeffs, cstride, perms);
    values.emplace(std::make_pair(IntegralType::exterior_facet, i), std::span<T>(scalars));
  }

  for (int i : M.integral_ids(IntegralType::interior_facet))
  {
    const std::vector<int> c_offsets = M.coefficient_offsets();
    auto fn = M.kernel(IntegralType::interior_facet, i);
    assert(fn);
    auto& [coeffs, cstride] = coefficients.at({IntegralType::interior_facet, i});
    std::span<const std::int32_t> interior_facets = M.domain(IntegralType::interior_facet, i);

    std::vector<T> scalars(interior_facets.size());
    impl::assemble_interior_facets<T>(std::span<T>(scalars), x_dofmap, x, num_facets_per_cell, interior_facets,
                                      fn, constants, coeffs, cstride, c_offsets, perms);
    values.emplace(std::make_pair(IntegralType::interior_facet, i), std::span<T>(scalars));
  }
  return values;
}

} // namespace dolfinx::fem::impl
