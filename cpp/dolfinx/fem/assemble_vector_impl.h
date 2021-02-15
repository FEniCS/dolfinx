// Copyright (C) 2018-2019 Garth N. Wells
//
// This file is part of DOLFINX (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include "DirichletBC.h"
#include "DofMap.h"
#include "Form.h"
#include "utils.h"
#include <dolfinx/common/IndexMap.h>
#include <dolfinx/common/types.h>
#include <dolfinx/fem/Constant.h>
#include <dolfinx/fem/FunctionSpace.h>
#include <dolfinx/graph/AdjacencyList.h>
#include <dolfinx/mesh/Geometry.h>
#include <dolfinx/mesh/Mesh.h>
#include <dolfinx/mesh/Topology.h>
#include <functional>
#include <memory>
#include <vector>

namespace dolfinx::fem::impl
{

/// Implementation of assembly

/// Assemble linear form into a vector
/// @param[in,out] b The vector to be assembled. It will not be zeroed
/// before assembly.
/// @param[in] L The linear forms to assemble into b
template <typename T>
void assemble_vector(tcb::span<T> b, const Form<T>& L);

/// Execute kernel over cells and accumulate result in vector
template <typename T>
void assemble_cells(
    tcb::span<T> b, const mesh::Geometry& geometry,
    const std::vector<std::int32_t>& active_cells,
    const graph::AdjacencyList<std::int32_t>& dofmap, const int bs,
    const std::function<void(T*, const T*, const T*, const double*, const int*,
                             const std::uint8_t*, const std::uint32_t)>& kernel,
    const common::array2d<T>& coeffs, const std::vector<T>& constant_values,
    const std::vector<std::uint32_t>& cell_info);

/// Execute kernel over cells and accumulate result in vector
template <typename T>
void assemble_exterior_facets(
    tcb::span<T> b, const mesh::Mesh& mesh,
    const std::vector<std::int32_t>& active_facets,
    const graph::AdjacencyList<std::int32_t>& dofmap, const int bs,
    const std::function<void(T*, const T*, const T*, const double*, const int*,
                             const std::uint8_t*, const std::uint32_t)>& fn,
    const common::array2d<T>& coeffs, const std::vector<T>& constant_values,
    const std::vector<std::uint32_t>& cell_info,
    const std::vector<std::uint8_t>& perms);

/// Assemble linear form interior facet integrals into an vector
template <typename T>
void assemble_interior_facets(
    tcb::span<T> b, const mesh::Mesh& mesh,
    const std::vector<std::int32_t>& active_facets, const fem::DofMap& dofmap,
    const std::function<void(T*, const T*, const T*, const double*, const int*,
                             const std::uint8_t*, const std::uint32_t)>& fn,
    const common::array2d<T>& coeffs, const std::vector<int>& offsets,
    const std::vector<T>& constant_values,
    const std::vector<std::uint32_t>& cell_info,
    const std::vector<std::uint8_t>& perms);

/// Modify b such that:
///
///   b <- b - scale * A_j (g_j - x0_j)
///
/// where j is a block (nest) row index. For a non-blocked problem j = 0.
/// The boundary conditions bc1 are on the trial spaces V_j. The forms
/// in [a] must have the same test space as L (from which b was built),
/// but the trial space may differ. If x0 is not supplied, then it is
/// treated as zero.
/// @param[in,out] b The vector to be modified
/// @param[in] a The bilinear forms, where a[j] is the form that
/// generates A_j
/// @param[in] bcs1 List of boundary conditions for each block, i.e.
/// bcs1[2] are the boundary conditions applied to the columns of a[2] /
/// x0[2] block
/// @param[in] x0 The vectors used in the lifting
/// @param[in] scale Scaling to apply
template <typename T>
void apply_lifting(
    tcb::span<T> b, const std::vector<std::shared_ptr<const Form<T>>> a,
    const std::vector<std::vector<std::shared_ptr<const DirichletBC<T>>>>& bcs1,
    const tcb::span<const T>& x0, double scale);

/// Modify RHS vector to account for boundary condition such that: b <-
////
///     b - scale * A (x_bc - x0)
////
/// @param[in,out] b The vector to be modified
/// @param[in] a The bilinear form that generates A
/// @param[in] bc_values1 The boundary condition 'values'
/// @param[in] bc_markers1 The indices (columns of A, rows of x) to
/// which bcs belong
/// @param[in] x0 The array used in the lifting, typically a 'current
/// solution' in a Newton method
/// @param[in] scale Scaling to apply
template <typename T>
void lift_bc(tcb::span<T> b, const Form<T>& a,
             const tcb::span<const T>& bc_values1,
             const std::vector<bool>& bc_markers1, const tcb::span<const T>& x0,
             double scale);

// Implementation of bc application
template <typename T>
void _lift_bc_cells(
    tcb::span<T> b, const mesh::Geometry& geometry,
    const std::function<void(T*, const T*, const T*, const double*, const int*,
                             const std::uint8_t*, const std::uint32_t)>& kernel,
    const std::vector<std::int32_t>& active_cells,
    const graph::AdjacencyList<std::int32_t>& dofmap0, int bs0,
    const graph::AdjacencyList<std::int32_t>& dofmap1, int bs1,
    const common::array2d<T>& coeffs, const std::vector<T>& constant_values,
    const std::vector<std::uint32_t>& cell_info,
    const tcb::span<const T>& bc_values1, const std::vector<bool>& bc_markers1,
    const tcb::span<const T>& x0, double scale)
{
  // Prepare cell geometry
  const int gdim = geometry.dim();
  const graph::AdjacencyList<std::int32_t>& x_dofmap = geometry.dofmap();

  // FIXME: Add proper interface for num coordinate dofs
  const int num_dofs_g = x_dofmap.num_links(0);
  const common::array2d<double>& x_g = geometry.x();

  // Data structures used in bc application
  std::vector<double> coordinate_dofs(num_dofs_g * gdim);
  std::vector<T> Ae, be;

  for (std::int32_t c : active_cells)
  {
    // Get dof maps for cell
    auto dmap1 = dofmap1.links(c);

    // Check if bc is applied to cell
    bool has_bc = false;
    for (std::size_t j = 0; j < dmap1.size(); ++j)
    {
      for (int k = 0; k < bs1; ++k)
      {
        assert(bs1 * dmap1[j] + k < (int)bc_markers1.size());
        if (bc_markers1[bs1 * dmap1[j] + k])
        {
          has_bc = true;
          break;
        }
      }
    }

    if (!has_bc)
      continue;

    // Get cell coordinates/geometry
    auto x_dofs = x_dofmap.links(c);
    for (std::size_t i = 0; i < x_dofs.size(); ++i)
    {
      std::copy_n(x_g.row(x_dofs[i]).data(), gdim,
                  std::next(coordinate_dofs.begin(), i * gdim));
    }

    // Size data structure for assembly
    auto dmap0 = dofmap0.links(c);

    const int num_rows = bs0 * dmap0.size();
    const int num_cols = bs1 * dmap1.size();

    auto coeff_array = coeffs.row(c);
    Ae.resize(num_rows * num_cols);
    std::fill(Ae.begin(), Ae.end(), 0);
    kernel(Ae.data(), coeff_array.data(), constant_values.data(),
           coordinate_dofs.data(), nullptr, nullptr, cell_info[c]);

    // Size data structure for assembly
    be.resize(num_rows);
    std::fill(be.begin(), be.end(), 0);
    for (std::size_t j = 0; j < dmap1.size(); ++j)
    {
      for (int k = 0; k < bs1; ++k)
      {
        const std::int32_t jj = bs1 * dmap1[j] + k;
        assert(jj < (int)bc_markers1.size());
        if (bc_markers1[jj])
        {
          const T bc = bc_values1[jj];
          const T _x0 = x0.empty() ? 0.0 : x0[jj];
          // be -= Ae.col(bs1 * j + k) * scale * (bc - _x0);
          for (int m = 0; m < num_rows; ++m)
            be[m] -= Ae[m * num_cols + bs1 * j + k] * scale * (bc - _x0);
        }
      }
    }

    for (std::size_t i = 0; i < dmap0.size(); ++i)
      for (int k = 0; k < bs0; ++k)
        b[bs0 * dmap0[i] + k] += be[bs0 * i + k];
  }
}
//----------------------------------------------------------------------------
template <typename T>
void _lift_bc_exterior_facets(
    tcb::span<T> b, const mesh::Mesh& mesh,
    const std::function<void(T*, const T*, const T*, const double*, const int*,
                             const std::uint8_t*, const std::uint32_t)>& kernel,
    const std::vector<std::int32_t>& active_facets,
    const graph::AdjacencyList<std::int32_t>& dofmap0, int bs0,
    const graph::AdjacencyList<std::int32_t>& dofmap1, int bs1,
    const common::array2d<T>& coeffs, const std::vector<T>& constant_values,
    const std::vector<std::uint32_t>& cell_info,
    const std::vector<std::uint8_t>& perms,
    const tcb::span<const T>& bc_values1, const std::vector<bool>& bc_markers1,
    const tcb::span<const T>& x0, double scale)
{
  const int gdim = mesh.geometry().dim();
  const int tdim = mesh.topology().dim();

  // Prepare cell geometry
  const graph::AdjacencyList<std::int32_t>& x_dofmap = mesh.geometry().dofmap();

  // FIXME: Add proper interface for num coordinate dofs
  const int num_dofs_g = x_dofmap.num_links(0);
  const common::array2d<double>& x_g = mesh.geometry().x();

  // Data structures used in bc application
  std::vector<double> coordinate_dofs(num_dofs_g * gdim);
  std::vector<T> Ae, be;

  // Iterate over owned facets
  const mesh::Topology& topology = mesh.topology();
  auto connectivity = topology.connectivity(tdim - 1, tdim);
  assert(connectivity);
  auto c_to_f = topology.connectivity(tdim, tdim - 1);
  assert(c_to_f);
  auto map = topology.index_map(tdim - 1);
  assert(map);

  for (std::int32_t f : active_facets)
  {
    // Create attached cell
    assert(connectivity->num_links(f) == 1);
    const std::int32_t cell = connectivity->links(f)[0];

    // Get local index of facet with respect to the cell
    auto facets = c_to_f->links(cell);
    auto it = std::find(facets.begin(), facets.end(), f);
    assert(it != facets.end());
    const int local_facet = std::distance(facets.begin(), it);

    // Get dof maps for cell
    auto dmap1 = dofmap1.links(cell);

    // Check if bc is applied to cell
    bool has_bc = false;
    for (std::size_t j = 0; j < dmap1.size(); ++j)
    {
      for (int k = 0; k < bs1; ++k)
      {
        if (bc_markers1[bs1 * dmap1[j] + k])
        {
          has_bc = true;
          break;
        }
      }
    }

    if (!has_bc)
      continue;

    // Get cell coordinates/geometry
    auto x_dofs = x_dofmap.links(cell);
    for (std::size_t i = 0; i < x_dofs.size(); ++i)
    {
      std::copy_n(x_g.row(x_dofs[i]).data(), gdim,
                  std::next(coordinate_dofs.begin(), i * gdim));
    }

    // Size data structure for assembly
    auto dmap0 = dofmap0.links(cell);

    const int num_rows = bs0 * dmap0.size();
    const int num_cols = bs1 * dmap1.size();

    auto coeff_array = coeffs.row(cell);
    Ae.resize(num_rows * num_cols);
    std::fill(Ae.begin(), Ae.end(), 0);
    kernel(Ae.data(), coeff_array.data(), constant_values.data(),
           coordinate_dofs.data(), &local_facet,
           &perms[cell * facets.size() + local_facet], cell_info[cell]);

    // Size data structure for assembly
    be.resize(num_rows);
    std::fill(be.begin(), be.end(), 0);
    for (std::size_t j = 0; j < dmap1.size(); ++j)
    {
      for (int k = 0; k < bs1; ++k)
      {
        const std::int32_t jj = bs1 * dmap1[j] + k;
        if (bc_markers1[jj])
        {

          const T bc = bc_values1[jj];
          const T _x0 = x0.empty() ? 0.0 : x0[jj];
          // be -= Ae.col(bs1 * j + k) * scale * (bc - _x0);
          for (int m = 0; m < num_rows; ++m)
            be[m] -= Ae[m * num_cols + bs1 * j + k] * scale * (bc - _x0);
        }
      }
    }

    for (std::size_t i = 0; i < dmap0.size(); ++i)
      for (int k = 0; k < bs0; ++k)
        b[bs0 * dmap0[i] + k] += be[bs0 * i + k];
  }
}
//----------------------------------------------------------------------------
template <typename T>
void _lift_bc_interior_facets(
    tcb::span<T> b, const mesh::Mesh& mesh,
    const std::function<void(T*, const T*, const T*, const double*, const int*,
                             const std::uint8_t*, const std::uint32_t)>& kernel,
    const std::vector<std::int32_t>& active_facets,
    const graph::AdjacencyList<std::int32_t>& dofmap0, int bs0,
    const graph::AdjacencyList<std::int32_t>& dofmap1, int bs1,
    const common::array2d<T>& coeffs, const std::vector<int>& offsets,
    const std::vector<T>& constant_values,
    const std::vector<std::uint32_t>& cell_info,
    const std::vector<std::uint8_t>& perms,
    const tcb::span<const T>& bc_values1, const std::vector<bool>& bc_markers1,
    const tcb::span<const T>& x0, double scale)
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
  std::vector<T> coeff_array(2 * offsets.back());
  assert(offsets.back() == int(coeffs.shape[1]));
  std::vector<T> Ae, be;

  // Temporaries for joint dofmaps
  std::vector<std::int32_t> dmapjoint0, dmapjoint1;

  const mesh::Topology& topology = mesh.topology();
  auto connectivity = topology.connectivity(tdim - 1, tdim);
  assert(connectivity);
  auto c_to_f = topology.connectivity(tdim, tdim - 1);
  assert(c_to_f);
  auto f_to_c = topology.connectivity(tdim - 1, tdim);
  assert(f_to_c);
  auto map = topology.index_map(tdim - 1);
  assert(map);

  const int offset_g = gdim * num_dofs_g;
  for (std::int32_t f : active_facets)
  {
    // Create attached cells
    auto cells = f_to_c->links(f);
    assert(cells.size() == 2);

    // Get local index of facet with respect to the cell
    auto facets0 = c_to_f->links(cells[0]);
    auto it0 = std::find(facets0.begin(), facets0.end(), f);
    assert(it0 != facets0.end());
    const int local_facet0 = std::distance(facets0.begin(), it0);
    auto facets1 = c_to_f->links(cells[1]);
    auto it1 = std::find(facets1.begin(), facets1.end(), f);
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
    const tcb::span<const std::int32_t> dmap0_cell0 = dofmap0.links(cells[0]);
    const tcb::span<const std::int32_t> dmap0_cell1 = dofmap0.links(cells[1]);
    dmapjoint0.resize(dmap0_cell0.size() + dmap0_cell1.size());
    std::copy(dmap0_cell0.begin(), dmap0_cell0.end(), dmapjoint0.begin());
    std::copy(dmap0_cell1.begin(), dmap0_cell1.end(),
              std::next(dmapjoint0.begin(), dmap0_cell0.size()));

    const tcb::span<const std::int32_t> dmap1_cell0 = dofmap1.links(cells[0]);
    const tcb::span<const std::int32_t> dmap1_cell1 = dofmap1.links(cells[1]);
    dmapjoint1.resize(dmap1_cell0.size() + dmap1_cell1.size());
    std::copy(dmap1_cell0.begin(), dmap1_cell0.end(), dmapjoint1.begin());
    std::copy(dmap1_cell1.begin(), dmap1_cell1.end(),
              std::next(dmapjoint1.begin(), dmap1_cell0.size()));

    // Check if bc is applied to cell0
    bool has_bc = false;
    for (std::size_t j = 0; j < dmap1_cell0.size(); ++j)
    {
      for (int k = 0; k < bs1; ++k)
      {
        if (bc_markers1[bs1 * dmap1_cell0[j] + k])
        {
          has_bc = true;
          break;
        }
      }
    }

    // Check if bc is applied to cell1
    for (std::size_t j = 0; j < dmap1_cell1.size(); ++j)
    {
      for (int k = 0; k < bs1; ++k)
      {
        if (bc_markers1[bs1 * dmap1_cell1[j] + k])
        {
          has_bc = true;
          break;
        }
      }
    }

    if (!has_bc)
      continue;

    // Layout for the restricted coefficients is flattened
    // w[coefficient][restriction][dof]
    const auto coeff_cell0 = coeffs.row(cells[0]);
    const auto coeff_cell1 = coeffs.row(cells[1]);

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
    kernel(Ae.data(), coeff_array.data(), constant_values.data(),
           coordinate_dofs.data(), local_facet.data(), perm.data(),
           cell_info[cells[0]]);

    be.resize(num_rows);
    std::fill(be.begin(), be.end(), 0);

    // Compute b = b - A*b for cell0
    for (std::size_t j = 0; j < dmap1_cell0.size(); ++j)
    {
      for (int k = 0; k < bs1; ++k)
      {
        const std::int32_t jj = bs1 * dmap1_cell0[j] + k;
        if (bc_markers1[jj])
        {
          const T bc = bc_values1[jj];
          const T _x0 = x0.empty() ? 0.0 : x0[jj];
          // be -= Ae.col(bs1 * j + k) * scale * (bc - _x0);
          for (int m = 0; m < num_rows; ++m)
            be[m] -= Ae[m * num_cols + bs1 * j + k] * scale * (bc - _x0);
        }
      }
    }

    // Compute b = b - A*b for cell1
    const int offset = bs1 * dmap1_cell0.size();
    for (std::size_t j = 0; j < dmap1_cell1.size(); ++j)
    {
      for (int k = 0; k < bs1; ++k)
      {
        const std::int32_t jj = bs1 * dmap1_cell1[j] + k;
        if (bc_markers1[jj])
        {
          const T bc = bc_values1[jj];
          const T _x0 = x0.empty() ? 0.0 : x0[jj];
          // be -= Ae.col(offset + bs1 * j + k) * scale * (bc - x0[jj]);
          for (int m = 0; m < num_rows; ++m)
          {
            be[m]
                -= Ae[m * num_cols + offset + bs1 * j + k] * scale * (bc - _x0);
          }
        }
      }
    }

    for (std::size_t i = 0; i < dmap0_cell0.size(); ++i)
      for (int k = 0; k < bs0; ++k)
        b[bs0 * dmap0_cell0[i] + k] += be[bs0 * i + k];

    for (std::size_t i = 0; i < dmap0_cell1.size(); ++i)
      for (int k = 0; k < bs0; ++k)
        b[bs0 * dmap0_cell1[i] + k] += be[offset + bs0 * i + k];
  }
}
//-----------------------------------------------------------------------------
template <typename T>
void assemble_vector(tcb::span<T> b, const Form<T>& L)
{
  std::shared_ptr<const mesh::Mesh> mesh = L.mesh();
  assert(mesh);
  const int tdim = mesh->topology().dim();
  const std::int32_t num_cells
      = mesh->topology().connectivity(tdim, 0)->num_nodes();

  // Get dofmap data
  assert(L.function_spaces().at(0));
  std::shared_ptr<const fem::DofMap> dofmap
      = L.function_spaces().at(0)->dofmap();
  assert(dofmap);
  const graph::AdjacencyList<std::int32_t>& dofs = dofmap->list();
  const int bs = dofmap->bs();

  // Prepare constants
  const std::vector<T> constant_values = pack_constants(L);

  // Prepare coefficients
  const common::array2d<T> coeffs = pack_coefficients(L);

  const bool needs_permutation_data = L.needs_permutation_data();
  if (needs_permutation_data)
    mesh->topology_mutable().create_entity_permutations();
  const std::vector<std::uint32_t>& cell_info
      = needs_permutation_data ? mesh->topology().get_cell_permutation_info()
                               : std::vector<std::uint32_t>(num_cells);

  for (int i : L.integral_ids(IntegralType::cell))
  {
    const auto& fn = L.kernel(IntegralType::cell, i);
    const std::vector<std::int32_t>& active_cells
        = L.domains(IntegralType::cell, i);
    fem::impl::assemble_cells(b, mesh->geometry(), active_cells, dofs, bs, fn,
                              coeffs, constant_values, cell_info);
  }

  if (L.num_integrals(IntegralType::exterior_facet) > 0
      or L.num_integrals(IntegralType::interior_facet) > 0)
  {
    // FIXME: cleanup these calls? Some of the happen internally again.
    mesh->topology_mutable().create_entities(tdim - 1);
    mesh->topology_mutable().create_connectivity(tdim - 1, tdim);
    mesh->topology_mutable().create_entity_permutations();

    const std::vector<std::uint8_t>& perms
        = mesh->topology().get_facet_permutations();
    for (int i : L.integral_ids(IntegralType::exterior_facet))
    {
      const auto& fn = L.kernel(IntegralType::exterior_facet, i);
      const std::vector<std::int32_t>& active_facets
          = L.domains(IntegralType::exterior_facet, i);
      fem::impl::assemble_exterior_facets(b, *mesh, active_facets, dofs, bs, fn,
                                          coeffs, constant_values, cell_info,
                                          perms);
    }

    const std::vector<int> c_offsets = L.coefficient_offsets();
    for (int i : L.integral_ids(IntegralType::interior_facet))
    {
      const auto& fn = L.kernel(IntegralType::interior_facet, i);
      const std::vector<std::int32_t>& active_facets
          = L.domains(IntegralType::interior_facet, i);
      fem::impl::assemble_interior_facets(b, *mesh, active_facets, *dofmap, fn,
                                          coeffs, c_offsets, constant_values,
                                          cell_info, perms);
    }
  }
}
//-----------------------------------------------------------------------------
template <typename T>
void assemble_cells(
    tcb::span<T> b, const mesh::Geometry& geometry,
    const std::vector<std::int32_t>& active_cells,
    const graph::AdjacencyList<std::int32_t>& dofmap, const int bs,
    const std::function<void(T*, const T*, const T*, const double*, const int*,
                             const std::uint8_t*, const std::uint32_t)>& kernel,
    const common::array2d<T>& coeffs, const std::vector<T>& constant_values,
    const std::vector<std::uint32_t>& cell_info)
{
  const int gdim = geometry.dim();

  // Prepare cell geometry
  const graph::AdjacencyList<std::int32_t>& x_dofmap = geometry.dofmap();

  // FIXME: Add proper interface for num coordinate dofs
  const int num_dofs_g = x_dofmap.num_links(0);
  const common::array2d<double>& x_g = geometry.x();

  // FIXME: Add proper interface for num_dofs
  // Create data structures used in assembly
  const int num_dofs = dofmap.links(0).size();
  std::vector<double> coordinate_dofs(num_dofs_g * gdim);
  std::vector<T> be(bs * num_dofs);

  // Iterate over active cells
  for (std::int32_t c : active_cells)
  {
    // Get cell coordinates/geometry
    auto x_dofs = x_dofmap.links(c);
    for (std::size_t i = 0; i < x_dofs.size(); ++i)
    {
      std::copy_n(x_g.row(x_dofs[i]).data(), gdim,
                  std::next(coordinate_dofs.begin(), i * gdim));
    }

    // Tabulate vector for cell
    std::fill(be.begin(), be.end(), 0);
    kernel(be.data(), coeffs.row(c).data(), constant_values.data(),
           coordinate_dofs.data(), nullptr, nullptr, cell_info[c]);

    // Scatter cell vector to 'global' vector array
    auto dofs = dofmap.links(c);
    for (int i = 0; i < num_dofs; ++i)
      for (int k = 0; k < bs; ++k)
        b[bs * dofs[i] + k] += be[bs * i + k];
  }
}
//-----------------------------------------------------------------------------
template <typename T>
void assemble_exterior_facets(
    tcb::span<T> b, const mesh::Mesh& mesh,
    const std::vector<std::int32_t>& active_facets,
    const graph::AdjacencyList<std::int32_t>& dofmap, const int bs,
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

  // FIXME: Add proper interface for num_dofs
  // Create data structures used in assembly
  const int num_dofs = dofmap.links(0).size();
  std::vector<double> coordinate_dofs(num_dofs_g * gdim);
  std::vector<T> be(bs * num_dofs);

  auto f_to_c = mesh.topology().connectivity(tdim - 1, tdim);
  assert(f_to_c);
  auto c_to_f = mesh.topology().connectivity(tdim, tdim - 1);
  assert(c_to_f);
  for (const auto& f : active_facets)
  {
    // Get index of first attached cell
    assert(f_to_c->num_links(f) > 0);
    const std::int32_t cell = f_to_c->links(f)[0];

    // Get local index of facet with respect to the cell
    auto facets = c_to_f->links(cell);
    auto it = std::find(facets.begin(), facets.end(), f);
    assert(it != facets.end());
    const int local_facet = std::distance(facets.begin(), it);

    // Get cell coordinates/geometry
    auto x_dofs = x_dofmap.links(cell);
    for (std::size_t i = 0; i < x_dofs.size(); ++i)
    {
      std::copy_n(x_g.row(x_dofs[i]).data(), gdim,
                  std::next(coordinate_dofs.begin(), i * gdim));
    }

    // Tabulate element vector
    std::fill(be.begin(), be.end(), 0);
    fn(be.data(), coeffs.row(cell).data(), constant_values.data(),
       coordinate_dofs.data(), &local_facet,
       &perms[cell * facets.size() + local_facet], cell_info[cell]);

    // Add element vector to global vector
    auto dofs = dofmap.links(cell);
    for (int i = 0; i < num_dofs; ++i)
      for (int k = 0; k < bs; ++k)
        b[bs * dofs[i] + k] += be[bs * i + k];
  }
}
//-----------------------------------------------------------------------------
template <typename T>
void assemble_interior_facets(
    tcb::span<T> b, const mesh::Mesh& mesh,
    const std::vector<std::int32_t>& active_facets, const fem::DofMap& dofmap,
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

  // Create data structures used in assembly
  std::vector<double> coordinate_dofs(2 * num_dofs_g * gdim);
  std::vector<T> be;
  std::vector<T> coeff_array(2 * offsets.back());
  assert(offsets.back() == int(coeffs.shape[1]));

  const int bs = dofmap.bs();
  auto f_to_c = mesh.topology().connectivity(tdim - 1, tdim);
  assert(f_to_c);
  auto c_to_f = mesh.topology().connectivity(tdim, tdim - 1);
  assert(c_to_f);
  const int offset_g = gdim * num_dofs_g;
  for (const auto& f : active_facets)
  {
    // Get attached cell indices
    auto cells = f_to_c->links(f);
    assert(cells.size() == 2);

    const int facets_per_cell = c_to_f->num_links(cells[0]);

    // Create attached cells
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

    // Get dofmaps for cells
    tcb::span<const std::int32_t> dmap0 = dofmap.cell_dofs(cells[0]);
    tcb::span<const std::int32_t> dmap1 = dofmap.cell_dofs(cells[1]);

    // Tabulate element vector
    be.resize(bs * (dmap0.size() + dmap1.size()));
    std::fill(be.begin(), be.end(), 0);
    const std::array perm{perms[cells[0] * facets_per_cell + local_facet[0]],
                          perms[cells[1] * facets_per_cell + local_facet[1]]};
    fn(be.data(), coeff_array.data(), constant_values.data(),
       coordinate_dofs.data(), local_facet.data(), perm.data(),
       cell_info[cells[0]]);

    // Add element vector to global vector
    for (std::size_t i = 0; i < dmap0.size(); ++i)
      for (int k = 0; k < bs; ++k)
        b[bs * dmap0[i] + k] += be[bs * i + k];
    for (std::size_t i = 0; i < dmap1.size(); ++i)
      for (int k = 0; k < bs; ++k)
        b[bs * dmap1[i] + k] += be[bs * (i + dmap0.size()) + k];
  }
}
//-----------------------------------------------------------------------------
template <typename T>
void apply_lifting(
    tcb::span<T> b, const std::vector<std::shared_ptr<const Form<T>>> a,
    const std::vector<std::vector<std::shared_ptr<const DirichletBC<T>>>>& bcs1,
    const std::vector<tcb::span<const T>>& x0, double scale)
{
  // FIXME: make changes to reactivate this check
  if (!x0.empty() and x0.size() != a.size())
  {
    throw std::runtime_error(
        "Mismatch in size between x0 and bilinear form in assembler.");
  }

  if (a.size() != bcs1.size())
  {
    throw std::runtime_error(
        "Mismatch in size between a and bcs in assembler.");
  }

  for (std::size_t j = 0; j < a.size(); ++j)
  {
    std::vector<bool> bc_markers1;
    std::vector<T> bc_values1;
    if (a[j] and !bcs1[j].empty())
    {
      auto V1 = a[j]->function_spaces()[1];
      assert(V1);
      auto map1 = V1->dofmap()->index_map;
      const int bs1 = V1->dofmap()->index_map_bs();
      assert(map1);
      const int crange = bs1 * (map1->size_local() + map1->num_ghosts());
      bc_markers1.assign(crange, false);
      bc_values1.assign(crange, 0.0);
      for (const std::shared_ptr<const DirichletBC<T>>& bc : bcs1[j])
      {
        bc->mark_dofs(bc_markers1);
        bc->dof_values(bc_values1);
      }

      if (!x0.empty())
        lift_bc<T>(b, *a[j], bc_values1, bc_markers1, x0[j], scale);
      else
      {
        lift_bc<T>(b, *a[j], bc_values1, bc_markers1, tcb::span<const T>(),
                   scale);
      }
    }
  }
}
//-----------------------------------------------------------------------------
template <typename T>
void lift_bc(tcb::span<T> b, const Form<T>& a,
             const tcb::span<const T>& bc_values1,
             const std::vector<bool>& bc_markers1, const tcb::span<const T>& x0,
             double scale)
{
  std::shared_ptr<const mesh::Mesh> mesh = a.mesh();
  assert(mesh);
  const int tdim = mesh->topology().dim();
  const std::int32_t num_cells
      = mesh->topology().connectivity(tdim, 0)->num_nodes();

  // Get dofmap for columns and rows of a
  assert(a.function_spaces().at(0));
  assert(a.function_spaces().at(1));
  const graph::AdjacencyList<std::int32_t>& dofmap0
      = a.function_spaces()[0]->dofmap()->list();
  const int bs0 = a.function_spaces()[0]->dofmap()->bs();
  const graph::AdjacencyList<std::int32_t>& dofmap1
      = a.function_spaces()[1]->dofmap()->list();
  const int bs1 = a.function_spaces()[1]->dofmap()->bs();

  // Prepare constants
  const std::vector<T> constant_values = pack_constants(a);

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
    const auto& kernel = a.kernel(IntegralType::cell, i);
    const std::vector<std::int32_t>& active_cells
        = a.domains(IntegralType::cell, i);
    _lift_bc_cells(b, mesh->geometry(), kernel, active_cells, dofmap0, bs0,
                   dofmap1, bs1, coeffs, constant_values, cell_info, bc_values1,
                   bc_markers1, x0, scale);
  }

  if (a.num_integrals(IntegralType::exterior_facet) > 0
      or a.num_integrals(IntegralType::interior_facet) > 0)
  {
    // FIXME: cleanup these calls? Some of the happen internally again.
    mesh->topology_mutable().create_entities(tdim - 1);
    mesh->topology_mutable().create_connectivity(tdim - 1, tdim);
    mesh->topology_mutable().create_entity_permutations();

    const std::vector<std::uint8_t>& perms
        = mesh->topology().get_facet_permutations();
    for (int i : a.integral_ids(IntegralType::exterior_facet))
    {
      const auto& kernel = a.kernel(IntegralType::exterior_facet, i);
      const std::vector<std::int32_t>& active_facets
          = a.domains(IntegralType::exterior_facet, i);
      _lift_bc_exterior_facets(b, *mesh, kernel, active_facets, dofmap0, bs0,
                               dofmap1, bs1, coeffs, constant_values, cell_info,
                               perms, bc_values1, bc_markers1, x0, scale);
    }

    const std::vector<int> c_offsets = a.coefficient_offsets();
    for (int i : a.integral_ids(IntegralType::interior_facet))
    {
      const auto& kernel = a.kernel(IntegralType::interior_facet, i);
      const std::vector<std::int32_t>& active_facets
          = a.domains(IntegralType::interior_facet, i);
      _lift_bc_interior_facets(b, *mesh, kernel, active_facets, dofmap0, bs0,
                               dofmap1, bs1, coeffs, c_offsets, constant_values,
                               cell_info, perms, bc_values1, bc_markers1, x0,
                               scale);
    }
  }
}
//-----------------------------------------------------------------------------
} // namespace dolfinx::fem::impl
