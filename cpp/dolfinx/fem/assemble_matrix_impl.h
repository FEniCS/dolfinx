// Copyright (C) 2018-2019 Garth N. Wells
//
// This file is part of DOLFINX (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include "DofMap.h"
#include "Form.h"
#include "utils.h"
#include <dolfinx/fem/FunctionSpace.h>
#include <dolfinx/graph/AdjacencyList.h>
#include <dolfinx/la/utils.h>
#include <dolfinx/mesh/Geometry.h>
#include <dolfinx/mesh/Mesh.h>
#include <dolfinx/mesh/Topology.h>
#include <functional>
#include <iterator>
#include <vector>

namespace dolfinx::fem::impl
{

/// The matrix A must already be initialised. The matrix may be a proxy,
/// i.e. a view into a larger matrix, and assembly is performed using
/// local indices. Rows (bc0) and columns (bc1) with Dirichlet
/// conditions are zeroed. Markers (bc0 and bc1) can be empty if not bcs
/// are applied. Matrix is not finalised.

template <typename T>
void assemble_matrix(
    const std::function<int(std::int32_t, const std::int32_t*, std::int32_t,
                            const std::int32_t*, const T*)>& mat_set_values,
    const Form<T>& a, const std::vector<bool>& bc0,
    const std::vector<bool>& bc1);

/// Execute kernel over cells and accumulate result in matrix
template <typename T>
void assemble_cells(
    const std::function<int(std::int32_t, const std::int32_t*, std::int32_t,
                            const std::int32_t*, const T*)>& mat_set_values,
    const mesh::Geometry& geometry,
    const std::vector<std::int32_t>& active_cells,
    const graph::AdjacencyList<std::int32_t>& dofmap0, const int bs0,
    const graph::AdjacencyList<std::int32_t>& dofmap1, const int bs1,
    const std::vector<bool>& bc0, const std::vector<bool>& bc1,
    const std::function<void(T*, const T*, const T*, const double*, const int*,
                             const std::uint8_t*, const std::uint32_t)>& kernel,
    const common::array2d<T>& coeffs, const std::vector<T>& constants,
    const std::vector<std::uint32_t>& cell_info);

/// Execute kernel over exterior facets and  accumulate result in Mat
template <typename T>
void assemble_exterior_facets(
    const std::function<int(std::int32_t, const std::int32_t*, std::int32_t,
                            const std::int32_t*, const T*)>& mat_set_values,
    const mesh::Mesh& mesh, const std::vector<std::int32_t>& active_facets,
    const graph::AdjacencyList<std::int32_t>& dofmap0, int bs0,
    const graph::AdjacencyList<std::int32_t>& dofmap1, int bs1,
    const std::vector<bool>& bc0, const std::vector<bool>& bc1,
    const std::function<void(T*, const T*, const T*, const double*, const int*,
                             const std::uint8_t*, const std::uint32_t)>& fn,
    const common::array2d<T>& coeffs, const std::vector<T>& constants,
    const std::vector<std::uint32_t>& cell_info,
    const std::vector<std::uint8_t>& perms);

/// Execute kernel over interior facets and  accumulate result in Mat
template <typename T>
void assemble_interior_facets(
    const std::function<int(std::int32_t, const std::int32_t*, std::int32_t,
                            const std::int32_t*, const T*)>& mat_set_values,
    const mesh::Mesh& mesh, const std::vector<std::int32_t>& active_facets,
    const DofMap& dofmap0, int bs0, const DofMap& dofmap1, int bs1,
    const std::vector<bool>& bc0, const std::vector<bool>& bc1,
    const std::function<void(T*, const T*, const T*, const double*, const int*,
                             const std::uint8_t*, const std::uint32_t)>& kernel,
    const common::array2d<T>& coeffs, const std::vector<int>& offsets,
    const std::vector<T>& constants,
    const std::vector<std::uint32_t>& cell_info,
    const std::vector<std::uint8_t>& perms);

//-----------------------------------------------------------------------------
template <typename T>
void assemble_matrix(
    const std::function<int(std::int32_t, const std::int32_t*, std::int32_t,
                            const std::int32_t*, const T*)>& mat_set_values,
    const Form<T>& a, const std::vector<bool>& bc0,
    const std::vector<bool>& bc1)
{
  std::shared_ptr<const mesh::Mesh> mesh = a.mesh();
  assert(mesh);
  const int tdim = mesh->topology().dim();
  const std::int32_t num_cells
      = mesh->topology().connectivity(tdim, 0)->num_nodes();

  // Get dofmap data
  std::shared_ptr<const fem::DofMap> dofmap0
      = a.function_spaces().at(0)->dofmap();
  std::shared_ptr<const fem::DofMap> dofmap1
      = a.function_spaces().at(1)->dofmap();
  assert(dofmap0);
  assert(dofmap1);
  const graph::AdjacencyList<std::int32_t>& dofs0 = dofmap0->list();
  const int bs0 = dofmap0->bs();
  const graph::AdjacencyList<std::int32_t>& dofs1 = dofmap1->list();
  const int bs1 = dofmap1->bs();

  // Prepare constants
  const std::vector<T> constants = pack_constants(a);

  // Prepare coefficients
  const common::array2d<T> coeffs = pack_coefficients(a);

  const bool needs_permutation_data = a.needs_permutation_data();
  if (needs_permutation_data)
    mesh->topology_mutable().create_entity_permutations();
  const std::vector<std::uint32_t>& cell_info
      = needs_permutation_data ? mesh->topology().get_cell_permutation_info()
                               : std::vector<std::uint32_t>(num_cells);

  for (int i : a.integral_ids(IntegralType::cell))
  {
    const auto& fn = a.kernel(IntegralType::cell, i);
    const std::vector<std::int32_t>& active_cells
        = a.domains(IntegralType::cell, i);
    impl::assemble_cells<T>(mat_set_values, mesh->geometry(), active_cells,
                            dofs0, bs0, dofs1, bs1, bc0, bc1, fn, coeffs,
                            constants, cell_info);
  }

  if (a.num_integrals(IntegralType::exterior_facet) > 0
      or a.num_integrals(IntegralType::interior_facet) > 0)
  {
    mesh->topology_mutable().create_connectivity(tdim - 1, tdim);
    mesh->topology_mutable().create_entity_permutations();

    const std::vector<std::uint8_t>& perms
        = mesh->topology().get_facet_permutations();

    for (int i : a.integral_ids(IntegralType::exterior_facet))
    {
      const auto& fn = a.kernel(IntegralType::exterior_facet, i);
      const std::vector<std::int32_t>& active_facets
          = a.domains(IntegralType::exterior_facet, i);
      impl::assemble_exterior_facets<T>(mat_set_values, *mesh, active_facets,
                                        dofs0, bs0, dofs1, bs1, bc0, bc1, fn,
                                        coeffs, constants, cell_info, perms);
    }

    const std::vector<int> c_offsets = a.coefficient_offsets();
    for (int i : a.integral_ids(IntegralType::interior_facet))
    {
      const auto& fn = a.kernel(IntegralType::interior_facet, i);
      const std::vector<std::int32_t>& active_facets
          = a.domains(IntegralType::interior_facet, i);
      impl::assemble_interior_facets<T>(
          mat_set_values, *mesh, active_facets, *dofmap0, bs0, *dofmap1, bs1,
          bc0, bc1, fn, coeffs, c_offsets, constants, cell_info, perms);
    }
  }
}
//-----------------------------------------------------------------------------
template <typename T>
void assemble_cells(
    const std::function<int(std::int32_t, const std::int32_t*, std::int32_t,
                            const std::int32_t*, const T*)>& mat_set,
    const mesh::Geometry& geometry,
    const std::vector<std::int32_t>& active_cells,
    const graph::AdjacencyList<std::int32_t>& dofmap0, const int bs0,
    const graph::AdjacencyList<std::int32_t>& dofmap1, const int bs1,
    const std::vector<bool>& bc0, const std::vector<bool>& bc1,
    const std::function<void(T*, const T*, const T*, const double*, const int*,
                             const std::uint8_t*, const std::uint32_t)>& kernel,
    const common::array2d<T>& coeffs, const std::vector<T>& constants,
    const std::vector<std::uint32_t>& cell_info)
{
  const int gdim = geometry.dim();

  // Prepare cell geometry
  const graph::AdjacencyList<std::int32_t>& x_dofmap = geometry.dofmap();

  // FIXME: Add proper interface for num coordinate dofs
  const int num_dofs_g = x_dofmap.num_links(0);
  const common::array2d<double>& x_g = geometry.x();

  // Iterate over active cells
  const int num_dofs0 = dofmap0.links(0).size();
  const int num_dofs1 = dofmap1.links(0).size();
  const int ndim0 = bs0 * num_dofs0;
  const int ndim1 = bs1 * num_dofs1;
  std::vector<T> Ae(ndim0 * ndim1);
  std::vector<double> coordinate_dofs(num_dofs_g * gdim);
  for (std::int32_t c : active_cells)
  {
    // Get cell coordinates/geometry
    auto x_dofs = x_dofmap.links(c);
    for (std::size_t i = 0; i < x_dofs.size(); ++i)
    {
      std::copy_n(x_g.row(x_dofs[i]).data(), gdim,
                  std::next(coordinate_dofs.begin(), i * gdim));
    }

    // Tabulate tensor
    std::fill(Ae.begin(), Ae.end(), 0);
    kernel(Ae.data(), coeffs.row(c).data(), constants.data(),
           coordinate_dofs.data(), nullptr, nullptr, cell_info[c]);

    // Zero rows/columns for essential bcs
    auto dofs0 = dofmap0.links(c);
    auto dofs1 = dofmap1.links(c);
    if (!bc0.empty())
    {
      for (int i = 0; i < num_dofs0; ++i)
      {
        for (int k = 0; k < bs0; ++k)
        {
          if (bc0[bs0 * dofs0[i] + k])
          {
            // Zero row bs0 * i + k
            const int row = bs0 * i + k;
            std::fill_n(std::next(Ae.begin(), ndim1 * row), ndim1, 0.0);
          }
        }
      }
    }

    if (!bc1.empty())
    {
      for (int j = 0; j < num_dofs1; ++j)
      {
        for (int k = 0; k < bs1; ++k)
        {
          if (bc1[bs1 * dofs1[j] + k])
          {
            // Zero column bs1 * j + k
            const int col = bs1 * j + k;
            for (int row = 0; row < ndim0; ++row)
              Ae[row * ndim1 + col] = 0.0;
          }
        }
      }
    }

    mat_set(dofs0.size(), dofs0.data(), dofs1.size(), dofs1.data(), Ae.data());
  }
} // namespace dolfinx::fem::impl
//-----------------------------------------------------------------------------
template <typename T>
void assemble_exterior_facets(
    const std::function<int(std::int32_t, const std::int32_t*, std::int32_t,
                            const std::int32_t*, const T*)>& mat_set_values,
    const mesh::Mesh& mesh, const std::vector<std::int32_t>& active_facets,
    const graph::AdjacencyList<std::int32_t>& dofmap0, int bs0,
    const graph::AdjacencyList<std::int32_t>& dofmap1, int bs1,
    const std::vector<bool>& bc0, const std::vector<bool>& bc1,
    const std::function<void(T*, const T*, const T*, const double*, const int*,
                             const std::uint8_t*, const std::uint32_t)>& kernel,
    const common::array2d<T>& coeffs, const std::vector<T>& constants,
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

  // Data structures used in assembly
  std::vector<double> coordinate_dofs(num_dofs_g * gdim);
  const int num_dofs0 = dofmap0.links(0).size();
  const int num_dofs1 = dofmap1.links(0).size();
  const int ndim0 = bs0 * num_dofs0;
  const int ndim1 = bs1 * num_dofs1;
  std::vector<T> Ae(ndim0 * ndim1);

  // Iterate over all facets
  auto f_to_c = mesh.topology().connectivity(tdim - 1, tdim);
  assert(f_to_c);
  auto c_to_f = mesh.topology().connectivity(tdim, tdim - 1);
  assert(c_to_f);
  for (std::int32_t f : active_facets)
  {
    auto cells = f_to_c->links(f);
    assert(cells.size() == 1);

    // Get local index of facet with respect to the cell
    auto facets = c_to_f->links(cells[0]);
    auto it = std::find(facets.begin(), facets.end(), f);
    assert(it != facets.end());
    const int local_facet = std::distance(facets.begin(), it);

    // Get cell coordinates/geometry
    auto x_dofs = x_dofmap.links(cells[0]);
    for (std::size_t i = 0; i < x_dofs.size(); ++i)
    {
      std::copy_n(x_g.row(x_dofs[i]).data(), gdim,
                  std::next(coordinate_dofs.begin(), i * gdim));
    }

    // Tabulate tensor
    std::fill(Ae.begin(), Ae.end(), 0);
    kernel(Ae.data(), coeffs.row(cells[0]).data(), constants.data(),
           coordinate_dofs.data(), &local_facet,
           &perms[cells[0] * facets.size() + local_facet], cell_info[cells[0]]);

    // Zero rows/columns for essential bcs
    auto dofs0 = dofmap0.links(cells[0]);
    auto dofs1 = dofmap1.links(cells[0]);
    if (!bc0.empty())
    {
      for (int i = 0; i < num_dofs0; ++i)
      {
        for (int k = 0; k < bs0; ++k)
        {
          if (bc0[bs0 * dofs0[i] + k])
          {
            // Zero row bs0 * i + k
            const int row = bs0 * i + k;
            std::fill_n(std::next(Ae.begin(), ndim1 * row), ndim1, 0.0);
          }
        }
      }
    }
    if (!bc1.empty())
    {
      for (int j = 0; j < num_dofs1; ++j)
      {
        for (int k = 0; k < bs1; ++k)
        {
          if (bc1[bs1 * dofs1[j] + k])
          {
            // Zero column bs1 * j + k
            const int col = bs1 * j + k;
            for (int row = 0; row < ndim0; ++row)
              Ae[row * ndim1 + col] = 0.0;
          }
        }
      }
    }

    mat_set_values(dofs0.size(), dofs0.data(), dofs1.size(), dofs1.data(),
                   Ae.data());
  }
}
//-----------------------------------------------------------------------------
template <typename T>
void assemble_interior_facets(
    const std::function<int(std::int32_t, const std::int32_t*, std::int32_t,
                            const std::int32_t*, const T*)>& mat_set_values,
    const mesh::Mesh& mesh, const std::vector<std::int32_t>& active_facets,
    const DofMap& dofmap0, int bs0, const DofMap& dofmap1, int bs1,
    const std::vector<bool>& bc0, const std::vector<bool>& bc1,
    const std::function<void(T*, const T*, const T*, const double*, const int*,
                             const std::uint8_t*, const std::uint32_t)>& fn,
    const common::array2d<T>& coeffs, const std::vector<int>& offsets,
    const std::vector<T>& constants,
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

  // Data structures used in assembly
  std::vector<double> coordinate_dofs(2 * num_dofs_g * gdim);
  std::vector<T> Ae, be;
  std::vector<T> coeff_array(2 * offsets.back());
  assert(offsets.back() == coeffs.shape[1]);

  // Temporaries for joint dofmaps
  std::vector<std::int32_t> dmapjoint0, dmapjoint1;

  // Iterate over all facets
  auto c = mesh.topology().connectivity(tdim - 1, tdim);
  assert(c);
  auto c_to_f = mesh.topology().connectivity(tdim, tdim - 1);
  assert(c_to_f);
  const int offset_g = gdim * num_dofs_g;
  for (std::int32_t facet_index : active_facets)
  {
    // Create attached cells
    auto cells = c->links(facet_index);
    assert(cells.size() == 2);

    // Get local index of facet with respect to the cell
    auto facets0 = c_to_f->links(cells[0]);
    const auto* it0 = std::find(facets0.begin(), facets0.end(), facet_index);
    assert(it0 != facets0.end());
    const int local_facet0 = std::distance(facets0.begin(), it0);
    auto facets1 = c_to_f->links(cells[1]);
    const auto* it1 = std::find(facets1.begin(), facets1.end(), facet_index);
    assert(it1 != facets1.end());
    const int local_facet1 = std::distance(facets1.begin(), it1);

    const std::array local_facet{local_facet0, local_facet1};

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

    // Get dof maps for cells and pack
    tcb::span<const std::int32_t> dmap0_cell0 = dofmap0.cell_dofs(cells[0]);
    tcb::span<const std::int32_t> dmap0_cell1 = dofmap0.cell_dofs(cells[1]);
    dmapjoint0.resize(dmap0_cell0.size() + dmap0_cell1.size());
    std::copy(dmap0_cell0.begin(), dmap0_cell0.end(), dmapjoint0.begin());
    std::copy(dmap0_cell1.begin(), dmap0_cell1.end(),
              std::next(dmapjoint0.begin(), dmap0_cell0.size()));

    tcb::span<const std::int32_t> dmap1_cell0 = dofmap1.cell_dofs(cells[0]);
    tcb::span<const std::int32_t> dmap1_cell1 = dofmap1.cell_dofs(cells[1]);
    dmapjoint1.resize(dmap1_cell0.size() + dmap1_cell1.size());
    std::copy(dmap1_cell0.begin(), dmap1_cell0.end(), dmapjoint1.begin());
    std::copy(dmap1_cell1.begin(), dmap1_cell1.end(),
              std::next(dmapjoint1.begin(), dmap1_cell0.size()));

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

    const int num_rows = bs0 * dmapjoint0.size();
    const int num_cols = bs1 * dmapjoint1.size();

    // Tabulate tensor
    Ae.resize(num_rows * num_cols);
    std::fill(Ae.begin(), Ae.end(), 0);

    const int facets_per_cell = facets0.size();
    const std::array perm{perms[cells[0] * facets_per_cell + local_facet[0]],
                          perms[cells[1] * facets_per_cell + local_facet[1]]};
    fn(Ae.data(), coeff_array.data(), constants.data(), coordinate_dofs.data(),
       local_facet.data(), perm.data(), cell_info[cells[0]]);

    // Zero rows/columns for essential bcs
    if (!bc0.empty())
    {
      for (std::size_t i = 0; i < dmapjoint0.size(); ++i)
      {
        for (int k = 0; k < bs0; ++k)
        {
          if (bc0[bs0 * dmapjoint0[i] + k])
          {
            // Zero row bs0 * i + k
            std::fill_n(std::next(Ae.begin(), num_cols * (bs0 * i + k)),
                        num_cols, 0.0);
          }
        }
      }
    }
    if (!bc1.empty())
    {
      for (std::size_t j = 0; j < dmapjoint1.size(); ++j)
      {
        for (int k = 0; k < bs1; ++k)
        {
          if (bc1[bs1 * dmapjoint1[j] + k])
          {
            // Zero column bs1 * j + k
            for (int m = 0; m < num_rows; ++m)
              Ae[m * num_cols + bs1 * j + k] = 0.0;
          }
        }
      }
    }

    mat_set_values(dmapjoint0.size(), dmapjoint0.data(), dmapjoint1.size(),
                   dmapjoint1.data(), Ae.data());
  }
}
//-----------------------------------------------------------------------------

} // namespace dolfinx::fem::impl
