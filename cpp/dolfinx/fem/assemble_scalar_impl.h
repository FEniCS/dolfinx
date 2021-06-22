// Copyright (C) 2019-2020 Garth N. Wells
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
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

/// Assemble functional over cells
template <typename T>
T assemble_cells(const mesh::Geometry& geometry,
                 const xtl::span<const std::int32_t>& active_cells,
                 const std::function<void(T*, const T*, const T*, const double*,
                                          const int*, const std::uint8_t*)>& fn,
                 const xtl::span<const T>& constants, const array2d<T>& coeffs)
{
  // Prepare cell geometry
  const graph::AdjacencyList<std::int32_t>& x_dofmap = geometry.dofmap();

  // FIXME: Add proper interface for num coordinate dofs
  const std::size_t num_dofs_g = x_dofmap.num_links(0);
  const xt::xtensor<double, 2>& x_g = geometry.x();

  // Create data structures used in assembly
  std::vector<double> coordinate_dofs(3 * num_dofs_g);

  // Iterate over all cells
  T value(0);
  for (std::int32_t c : active_cells)
  {
    // Get cell coordinates/geometry
    auto x_dofs = x_dofmap.links(c);
    for (std::size_t i = 0; i < x_dofs.size(); ++i)
    {
      std::copy_n(xt::row(x_g, x_dofs[i]).begin(), 3,
                  std::next(coordinate_dofs.begin(), 3 * i));
    }

    auto coeff_cell = coeffs.row(c);
    fn(&value, coeff_cell.data(), constants.data(), coordinate_dofs.data(),
       nullptr, nullptr);
  }

  return value;
}

/// Execute kernel over exterior facets and accumulate result
template <typename T>
T assemble_exterior_facets(
    const mesh::Mesh& mesh, const xtl::span<const std::int32_t>& active_facets,
    const std::function<void(T*, const T*, const T*, const double*, const int*,
                             const std::uint8_t*)>& fn,
    const xtl::span<const T>& constants, const array2d<T>& coeffs,
    const xtl::span<const std::uint8_t>& perms)
{
  const int tdim = mesh.topology().dim();

  // Prepare cell geometry
  const graph::AdjacencyList<std::int32_t>& x_dofmap = mesh.geometry().dofmap();

  // FIXME: Add proper interface for num coordinate dofs
  const std::size_t num_dofs_g = x_dofmap.num_links(0);
  const xt::xtensor<double, 2>& x_g = mesh.geometry().x();

  // Create data structures used in assembly
  std::vector<double> coordinate_dofs(3 * num_dofs_g);

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
      std::copy_n(xt::row(x_g, x_dofs[i]).begin(), 3,
                  std::next(coordinate_dofs.begin(), 3 * i));
    }

    auto coeff_cell = coeffs.row(cell);
    fn(&value, coeff_cell.data(), constants.data(), coordinate_dofs.data(),
       &local_facet, &perms[cell * facets.size() + local_facet]);
  }

  return value;
}

/// Assemble functional over interior facets
template <typename T>
T assemble_interior_facets(
    const mesh::Mesh& mesh, const xtl::span<const std::int32_t>& active_facets,
    const std::function<void(T*, const T*, const T*, const double*, const int*,
                             const std::uint8_t*)>& fn,
    const xtl::span<const T>& constants, const array2d<T>& coeffs,
    const xtl::span<const int>& offsets,
    const xtl::span<const std::uint8_t>& perms)
{
  const int tdim = mesh.topology().dim();

  // Prepare cell geometry
  const graph::AdjacencyList<std::int32_t>& x_dofmap = mesh.geometry().dofmap();

  // FIXME: Add proper interface for num coordinate dofs
  const std::size_t num_dofs_g = x_dofmap.num_links(0);
  const xt::xtensor<double, 2>& x_g = mesh.geometry().x();

  // Create data structures used in assembly
  xt::xtensor<double, 3> coordinate_dofs({2, num_dofs_g, 3});
  std::vector<T> coeff_array(2 * offsets.back());
  assert(offsets.back() == coeffs.shape[1]);

  auto f_to_c = mesh.topology().connectivity(tdim - 1, tdim);
  assert(f_to_c);
  auto c_to_f = mesh.topology().connectivity(tdim, tdim - 1);
  assert(c_to_f);

  // Iterate over all facets
  T value = 0;
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
    for (std::size_t i = 0; i < x_dofs0.size(); ++i)
    {
      std::copy_n(xt::view(x_g, x_dofs0[i]).begin(), 3,
                  xt::view(coordinate_dofs, 0, i, xt::all()).begin());
    }
    auto x_dofs1 = x_dofmap.links(cells[1]);
    for (std::size_t i = 0; i < x_dofs1.size(); ++i)
    {
      std::copy_n(xt::view(x_g, x_dofs1[i]).begin(), 3,
                  xt::view(coordinate_dofs, 1, i, xt::all()).begin());
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
    fn(&value, coeff_array.data(), constants.data(), coordinate_dofs.data(),
       local_facet.data(), perm.data());
  }

  return value;
}

/// Assemble functional into an scalar
template <typename T>
T assemble_scalar(const fem::Form<T>& M, const xtl::span<const T>& constants,
                  const array2d<T>& coeffs)
{
  std::shared_ptr<const mesh::Mesh> mesh = M.mesh();
  assert(mesh);
  const int tdim = mesh->topology().dim();

  T value(0);
  for (int i : M.integral_ids(IntegralType::cell))
  {
    const auto& fn = M.kernel(IntegralType::cell, i);
    const std::vector<std::int32_t>& active_cells
        = M.domains(IntegralType::cell, i);
    value += impl::assemble_cells(mesh->geometry(), active_cells, fn, constants,
                                  coeffs);
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
      value += impl::assemble_exterior_facets(*mesh, active_facets, fn,
                                              constants, coeffs, perms);
    }

    const std::vector<int> c_offsets = M.coefficient_offsets();
    for (int i : M.integral_ids(IntegralType::interior_facet))
    {
      const auto& fn = M.kernel(IntegralType::interior_facet, i);
      const std::vector<std::int32_t>& active_facets
          = M.domains(IntegralType::interior_facet, i);
      value += impl::assemble_interior_facets(
          *mesh, active_facets, fn, constants, coeffs, c_offsets, perms);
    }
  }

  return value;
}

} // namespace dolfinx::fem::impl
