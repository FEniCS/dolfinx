// Copyright (C) 2019-2020 Garth N. Wells
//
// This file is part of DOLFINX (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include "Form.h"
#include "utils.h"
#include <dolfinx/common/IndexMap.h>
#include <dolfinx/common/types.h>
#include <dolfinx/fem/Constant.h>
#include <dolfinx/fem/FunctionSpace.h>
#include <dolfinx/mesh/Geometry.h>
#include <dolfinx/mesh/Mesh.h>
#include <dolfinx/mesh/Topology.h>
#include <memory>
#include <vector>

namespace dolfinx::fem::impl
{

/// Assemble functional into an scalar
template <typename T>
T assemble_scalar(const fem::Form<T>& M);

/// Assemble functional over cells
template <typename T>
T assemble_cells(
    const mesh::Geometry& geometry,
    const std::vector<std::int32_t>& active_cells,
    const std::function<void(T*, const T*, const T*, const double*, const int*,
                             const std::uint8_t*, const std::uint32_t)>& fn,
    const common::array2d<T>& coeffs, const std::vector<T>& constant_values,
    const std::vector<std::uint32_t>& cell_info);

/// Execute kernel over exterior facets and accumulate result
template <typename T>
T assemble_exterior_facets(
    const mesh::Mesh& mesh, const std::vector<std::int32_t>& active_cells,
    const std::function<void(T*, const T*, const T*, const double*, const int*,
                             const std::uint8_t*, const std::uint32_t)>& fn,
    const common::array2d<T>& coeffs, const std::vector<T>& constant_values,
    const std::vector<std::uint32_t>& cell_info,
    const std::vector<std::uint8_t>& perms);

/// Assemble functional over interior facets
template <typename T>
T assemble_interior_facets(
    const mesh::Mesh& mesh, const std::vector<std::int32_t>& active_cells,
    const std::function<void(T*, const T*, const T*, const double*, const int*,
                             const std::uint8_t*, const std::uint32_t)>& fn,
    const common::array2d<T>& coeffs, const std::vector<int>& offsets,
    const std::vector<T>& constant_values,
    const std::vector<std::uint32_t>& cell_info,
    const std::vector<std::uint8_t>& perms);

//-----------------------------------------------------------------------------
template <typename T>
T assemble_scalar(const fem::Form<T>& M)
{
  std::shared_ptr<const mesh::Mesh> mesh = M.mesh();
  assert(mesh);
  const int tdim = mesh->topology().dim();
  const std::int32_t num_cells
      = mesh->topology().connectivity(tdim, 0)->num_nodes();

  // Prepare constants
  const std::vector<std::shared_ptr<const fem::Constant<T>>>& constants
      = M.constants();

  std::vector<T> constant_values;
  for (auto const& constant : constants)
  {
    // Get underlying data array of this Constant
    const std::vector<T>& array = constant->value;
    constant_values.insert(constant_values.end(), array.begin(), array.end());
  }

  // Prepare coefficients
  const common::array2d<T> coeffs = pack_coefficients(M);

  const bool needs_permutation_data = M.needs_permutation_data();
  if (needs_permutation_data)
    mesh->topology_mutable().create_entity_permutations();
  const std::vector<std::uint32_t>& cell_info
      = needs_permutation_data ? mesh->topology().get_cell_permutation_info()
                               : std::vector<std::uint32_t>(num_cells);

  T value(0);
  for (int i : M.integral_ids(IntegralType::cell))
  {
    const auto& fn = M.kernel(IntegralType::cell, i);
    const std::vector<std::int32_t>& active_cells
        = M.domains(IntegralType::cell, i);
    value += fem::impl::assemble_cells(mesh->geometry(), active_cells, fn,
                                       coeffs, constant_values, cell_info);
  }

  if (M.num_integrals(IntegralType::exterior_facet) > 0
      or M.num_integrals(IntegralType::interior_facet) > 0)
  {
    // FIXME: cleanup these calls? Some of these happen internally again.
    mesh->topology_mutable().create_entities(tdim - 1);
    mesh->topology_mutable().create_connectivity(tdim - 1, tdim);
    mesh->topology_mutable().create_entity_permutations();

    const std::vector<std::uint8_t>& perms
        = mesh->topology().get_facet_permutations();

    for (int i : M.integral_ids(IntegralType::exterior_facet))
    {
      const auto& fn = M.kernel(IntegralType::exterior_facet, i);
      const std::vector<std::int32_t>& active_facets
          = M.domains(IntegralType::exterior_facet, i);
      value += fem::impl::assemble_exterior_facets(
          *mesh, active_facets, fn, coeffs, constant_values, cell_info, perms);
    }

    const std::vector<int> c_offsets = M.coefficient_offsets();
    for (int i : M.integral_ids(IntegralType::interior_facet))
    {
      const auto& fn = M.kernel(IntegralType::interior_facet, i);
      const std::vector<std::int32_t>& active_facets
          = M.domains(IntegralType::interior_facet, i);
      value += fem::impl::assemble_interior_facets(
          *mesh, active_facets, fn, coeffs, c_offsets, constant_values,
          cell_info, perms);
    }
  }

  return value;
}
//-----------------------------------------------------------------------------
template <typename T>
T assemble_cells(
    const mesh::Geometry& geometry,
    const std::vector<std::int32_t>& active_cells,
    const std::function<void(T*, const T*, const T*, const double*, const int*,
                             const std::uint8_t*, const std::uint32_t)>& fn,
    const common::array2d<T>& coeffs, const std::vector<T>& constant_values,
    const std::vector<std::uint32_t>& cell_info)
{
  const int gdim = geometry.dim();

  // Prepare cell geometry
  const graph::AdjacencyList<std::int32_t>& x_dofmap = geometry.dofmap();

  // FIXME: Add proper interface for num coordinate dofs
  const int num_dofs_g = x_dofmap.num_links(0);
  const common::array2d<double>& x_g = geometry.x();

  // Create data structures used in assembly
  std::vector<double> coordinate_dofs(num_dofs_g * gdim);

  // Iterate over all cells
  T value(0);
  for (std::int32_t c : active_cells)
  {
    // Get cell coordinates/geometry
    auto x_dofs = x_dofmap.links(c);
    for (std::size_t i = 0; i < x_dofs.size(); ++i)
    {
      std::copy_n(x_g.row(x_dofs[i]).data(), gdim,
                  std::next(coordinate_dofs.begin(), i * gdim));
    }

    auto coeff_cell = coeffs.row(c);
    fn(&value, coeff_cell.data(), constant_values.data(),
       coordinate_dofs.data(), nullptr, nullptr, cell_info[c]);
  }

  return value;
}
//-----------------------------------------------------------------------------
template <typename T>
T assemble_exterior_facets(
    const mesh::Mesh& mesh, const std::vector<std::int32_t>& active_facets,
    const std::function<void(T*, const T*, const T*, const double*, const int*,
                             const std::uint8_t*, const std::uint32_t)>& fn,
    const common::array2d<T>& coeffs, const std::vector<T>& constant_values,
    const std::vector<std::uint32_t>& cell_info,
    const std::vector<std::uint8_t>& perms)
{
  const int gdim = mesh.geometry().dim();
  const int tdim = mesh.topology().dim();

  // Prepare cell geometry
  const graph::AdjacencyList<std::int32_t>& x_dofmap = mesh.geometry().dofmap();

  // FIXME: Add proper interface for num coordinate dofs
  const int num_dofs_g = x_dofmap.num_links(0);
  const common::array2d<double>& x_g = mesh.geometry().x();

  // Creat data structures used in assembly
  std::vector<double> coordinate_dofs(num_dofs_g * gdim);

  auto f_to_c = mesh.topology().connectivity(tdim - 1, tdim);
  assert(f_to_c);
  auto c_to_f = mesh.topology().connectivity(tdim, tdim - 1);
  assert(c_to_f);

  // Iterate over all facets
  T value(0);
  for (std::int32_t facet : active_facets)
  {
    // Create attached cell
    assert(f_to_c->num_links(facet) == 1);
    const int cell = f_to_c->links(facet)[0];

    // Get local index of facet with respect to the cell
    auto facets = c_to_f->links(cell);
    auto it = std::find(facets.begin(), facets.end(), facet);
    assert(it != facets.end());
    const int local_facet = std::distance(facets.data(), it);

    // Get cell coordinates/geometry
    auto x_dofs = x_dofmap.links(cell);
    for (std::size_t i = 0; i < x_dofs.size(); ++i)
    {
      std::copy_n(x_g.row(x_dofs[i]).data(), gdim,
                  std::next(coordinate_dofs.begin(), i * gdim));
    }

    auto coeff_cell = coeffs.row(cell);
    fn(&value, coeff_cell.data(), constant_values.data(),
       coordinate_dofs.data(), &local_facet,
       &perms[cell * facets.size() + local_facet], cell_info[cell]);
  }

  return value;
}
//-----------------------------------------------------------------------------
template <typename T>
T assemble_interior_facets(
    const mesh::Mesh& mesh, const std::vector<std::int32_t>& active_facets,
    const std::function<void(T*, const T*, const T*, const double*, const int*,
                             const std::uint8_t*, const std::uint32_t)>& fn,
    const common::array2d<T>& coeffs, const std::vector<int>& offsets,
    const std::vector<T>& constant_values,
    const std::vector<std::uint32_t>& cell_info,
    const std::vector<std::uint8_t>& perms)
{
  const int gdim = mesh.geometry().dim();
  const int tdim = mesh.topology().dim();

  // Prepare cell geometry
  const graph::AdjacencyList<std::int32_t>& x_dofmap = mesh.geometry().dofmap();

  // FIXME: Add proper interface for num coordinate dofs
  const int num_dofs_g = x_dofmap.num_links(0);
  const common::array2d<double>& x_g = mesh.geometry().x();

  // Creat data structures used in assembly
  std::vector<double> coordinate_dofs(2 * num_dofs_g * gdim);
  std::vector<T> coeff_array(2 * offsets.back());
  assert(offsets.back() == coeffs.shape[1]);

  auto f_to_c = mesh.topology().connectivity(tdim - 1, tdim);
  assert(f_to_c);
  auto c_to_f = mesh.topology().connectivity(tdim, tdim - 1);
  assert(c_to_f);

  // Iterate over all facets
  T value = 0;
  const int offset_g = gdim * num_dofs_g;
  for (std::int32_t f : active_facets)
  {
    // Create attached cell
    auto cells = f_to_c->links(f);
    assert(cells.size() == 2);

    const int facets_per_cell = c_to_f->num_links(cells[0]);

    // Get local index of facet with respect to the cell
    std::array<int, 2> local_facet;
    for (int i = 0; i < 2; ++i)
    {
      auto facets = c_to_f->links(cells[i]);
      auto it = std::find(facets.begin(), facets.end(), f);
      assert(it != facets.end());
      local_facet[i] = std::distance(facets.begin(), it);
    }

    // Get cell geometry
    auto x_dofs0 = x_dofmap.links(cells[0]);
    auto x_dofs1 = x_dofmap.links(cells[1]);
    for (int i = 0; i < num_dofs_g; ++i)
    {
      for (int j = 0; j < gdim; ++j)
      {
        coordinate_dofs[i * gdim + j] = x_g(x_dofs0[i], j);
        coordinate_dofs[offset_g + i * gdim + j] = x_g(x_dofs1[i], j);
      }
    }

    // Layout for the restricted coefficients is flattened
    // w[coefficient][restriction][dof]
    auto coeff_cell0 = coeffs.row(cells[0]);
    auto coeff_cell1 = coeffs.row(cells[1]);

    // Loop over coefficients
    for (std::size_t i = 0; i < offsets.size() - 1; ++i)
    {
      // Loop over entries for coefficient i
      const int num_entries = offsets[i + 1] - offsets[i];
      std::copy_n(coeff_cell0.data() + offsets[i], num_entries,
                  std::next(coeff_array.begin(), 2 * offsets[i]));
      std::copy_n(coeff_cell1.data() + offsets[i], num_entries,
                  std::next(coeff_array.begin(), offsets[i + 1] + offsets[i]));
    }

    const std::array perm{perms[cells[0] * facets_per_cell + local_facet[0]],
                          perms[cells[1] * facets_per_cell + local_facet[1]]};
    fn(&value, coeff_array.data(), constant_values.data(),
       coordinate_dofs.data(), local_facet.data(), perm.data(),
       cell_info[cells[0]]);
  }

  return value;
}
//-----------------------------------------------------------------------------

} // namespace dolfinx::fem::impl
