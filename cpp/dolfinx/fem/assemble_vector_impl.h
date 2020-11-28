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
#include <Eigen/Dense>
#include <dolfinx/common/IndexMap.h>
#include <dolfinx/common/types.h>
#include <dolfinx/function/Constant.h>
#include <dolfinx/function/FunctionSpace.h>
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

/// Assemble linear form into an Eigen vector
/// @param[in,out] b The vector to be assembled. It will not be zeroed before
///   assembly.
/// @param[in] L The linear forms to assemble into b
template <typename T>
void assemble_vector(Eigen::Ref<Eigen::Matrix<T, Eigen::Dynamic, 1>> b,
                     const Form<T>& L);

/// Execute kernel over cells and accumulate result in vector
template <typename T>
void assemble_cells(
    Eigen::Ref<Eigen::Matrix<T, Eigen::Dynamic, 1>> b,
    const mesh::Geometry& geometry,
    const std::vector<std::int32_t>& active_cells,
    const graph::AdjacencyList<std::int32_t>& dofmap, const int bs,
    const std::function<void(T*, const T*, const T*, const double*, const int*,
                             const std::uint8_t*, const std::uint32_t)>& kernel,
    const Eigen::Array<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>&
        coeffs,
    const Eigen::Array<T, Eigen::Dynamic, 1>& constant_values,
    const Eigen::Array<std::uint32_t, Eigen::Dynamic, 1>& cell_info);

/// Execute kernel over cells and accumulate result in vector
template <typename T>
void assemble_exterior_facets(
    Eigen::Ref<Eigen::Matrix<T, Eigen::Dynamic, 1>> b, const mesh::Mesh& mesh,
    const std::vector<std::int32_t>& active_facets,
    const graph::AdjacencyList<std::int32_t>& dofmap, const int bs,
    const std::function<void(T*, const T*, const T*, const double*, const int*,
                             const std::uint8_t*, const std::uint32_t)>& fn,
    const Eigen::Array<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>&
        coeffs,
    const Eigen::Array<T, Eigen::Dynamic, 1>& constant_values,
    const Eigen::Array<std::uint32_t, Eigen::Dynamic, 1>& cell_info,
    const Eigen::Array<std::uint8_t, Eigen::Dynamic, Eigen::Dynamic>& perms);

/// Assemble linear form interior facet integrals into an Eigen vector
template <typename T>
void assemble_interior_facets(
    Eigen::Ref<Eigen::Matrix<T, Eigen::Dynamic, 1>> b, const mesh::Mesh& mesh,
    const std::vector<std::int32_t>& active_facets, const fem::DofMap& dofmap,
    const std::function<void(T*, const T*, const T*, const double*, const int*,
                             const std::uint8_t*, const std::uint32_t)>& fn,
    const Eigen::Array<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>&
        coeffs,
    const std::vector<int>& offsets,
    const Eigen::Array<T, Eigen::Dynamic, 1>& constant_values,
    const Eigen::Array<std::uint32_t, Eigen::Dynamic, 1>& cell_info,
    const Eigen::Array<std::uint8_t, Eigen::Dynamic, Eigen::Dynamic>& perms);

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
///              generates A_j
/// @param[in] bcs1 List of boundary conditions for each block, i.e.
///                 bcs1[2] are the boundary conditions applied to the
///                 columns of a[2] / x0[2] block
/// @param[in] x0 The vectors used in the lifting
/// @param[in] scale Scaling to apply
template <typename T>
void apply_lifting(
    Eigen::Ref<Eigen::Matrix<T, Eigen::Dynamic, 1>> b,
    const std::vector<std::shared_ptr<const Form<T>>> a,
    const std::vector<std::vector<std::shared_ptr<const DirichletBC<T>>>>& bcs1,
    const std::vector<Eigen::Ref<const Eigen::Matrix<T, Eigen::Dynamic, 1>>>&
        x0,
    double scale);

/// Modify RHS vector to account for boundary condition such that: b <-
////
///     b - scale * A (x_bc - x0)
////
/// @param[in,out] b The vector to be modified
/// @param[in] a The bilinear form that generates A
/// @param[in] bc_values1 The boundary condition 'values'
/// @param[in] bc_markers1 The indices (columns of A, rows of x) to
///                        which bcs belong
/// @param[in] x0 The array used in the lifting, typically a 'current
///               solution' in a Newton method
/// @param[in] scale Scaling to apply
template <typename T>
void lift_bc(
    Eigen::Ref<Eigen::Matrix<T, Eigen::Dynamic, 1>> b, const Form<T>& a,
    const Eigen::Ref<const Eigen::Matrix<T, Eigen::Dynamic, 1>>& bc_values1,
    const std::vector<bool>& bc_markers1,
    const Eigen::Ref<const Eigen::Matrix<T, Eigen::Dynamic, 1>>& x0,
    double scale);

// Implementation of bc application
template <typename T>
void _lift_bc_cells(
    Eigen::Ref<Eigen::Matrix<T, Eigen::Dynamic, 1>> b,
    const mesh::Geometry& geometry,
    const std::function<void(T*, const T*, const T*, const double*, const int*,
                             const std::uint8_t*, const std::uint32_t)>& kernel,
    const std::vector<std::int32_t>& active_cells,
    const graph::AdjacencyList<std::int32_t>& dofmap0, int bs0,
    const graph::AdjacencyList<std::int32_t>& dofmap1, int bs1,
    const Eigen::Array<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>&
        coeffs,
    const Eigen::Array<T, Eigen::Dynamic, 1>& constant_values,
    const Eigen::Array<std::uint32_t, Eigen::Dynamic, 1>& cell_info,
    const Eigen::Ref<const Eigen::Matrix<T, Eigen::Dynamic, 1>>& bc_values1,
    const std::vector<bool>& bc_markers1,
    const Eigen::Ref<const Eigen::Matrix<T, Eigen::Dynamic, 1>>& x0,
    double scale)
{
  // Prepare cell geometry
  const int gdim = geometry.dim();
  const graph::AdjacencyList<std::int32_t>& x_dofmap = geometry.dofmap();

  // FIXME: Add proper interface for num coordinate dofs
  const int num_dofs_g = x_dofmap.num_links(0);
  const Eigen::Array<double, Eigen::Dynamic, 3, Eigen::RowMajor>& x_g
      = geometry.x();

  // Data structures used in bc application
  Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
      coordinate_dofs(num_dofs_g, gdim);
  Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> Ae;
  Eigen::Matrix<T, Eigen::Dynamic, 1> be;

  for (std::int32_t c : active_cells)
  {
    // Get dof maps for cell
    auto dmap1 = dofmap1.links(c);

    // Check if bc is applied to cell
    bool has_bc = false;
    for (Eigen::Index j = 0; j < dmap1.size(); ++j)
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

    // Get cell vertex coordinates
    auto x_dofs = x_dofmap.links(c);
    for (int i = 0; i < num_dofs_g; ++i)
      for (int j = 0; j < gdim; ++j)
        coordinate_dofs(i, j) = x_g(x_dofs[i], j);

    // Size data structure for assembly
    auto dmap0 = dofmap0.links(c);

    auto coeff_array = coeffs.row(c);
    Ae.setZero(bs0 * dmap0.size(), bs1 * dmap1.size());
    kernel(Ae.data(), coeff_array.data(), constant_values.data(),
           coordinate_dofs.data(), nullptr, nullptr, cell_info[c]);

    // Size data structure for assembly
    be.setZero(bs0 * dmap0.size());
    for (Eigen::Index j = 0; j < dmap1.size(); ++j)
    {
      for (int k = 0; k < bs1; ++k)
      {
        const std::int32_t jj = bs1 * dmap1[j] + k;
        if (bc_markers1[jj])
        {
          const T bc = bc_values1[jj];
          if (x0.rows() > 0)
            be -= Ae.col(bs1 * j + k) * scale * (bc - x0[jj]);
          else
            be -= Ae.col(bs1 * j + k) * scale * bc;
        }
      }
    }

    for (Eigen::Index i = 0; i < dmap0.size(); ++i)
      for (int k = 0; k < bs0; ++k)
        b[bs0 * dmap0[i] + k] += be[bs0 * i + k];
  }
}
//----------------------------------------------------------------------------
template <typename T>
void _lift_bc_exterior_facets(
    Eigen::Ref<Eigen::Matrix<T, Eigen::Dynamic, 1>> b, const mesh::Mesh& mesh,
    const std::function<void(T*, const T*, const T*, const double*, const int*,
                             const std::uint8_t*, const std::uint32_t)>& kernel,
    const std::vector<std::int32_t>& active_facets,
    const graph::AdjacencyList<std::int32_t>& dofmap0, int bs0,
    const graph::AdjacencyList<std::int32_t>& dofmap1, int bs1,
    const Eigen::Array<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>&
        coeffs,
    const Eigen::Array<T, Eigen::Dynamic, 1>& constant_values,
    const Eigen::Array<std::uint32_t, Eigen::Dynamic, 1>& cell_info,
    const Eigen::Array<std::uint8_t, Eigen::Dynamic, Eigen::Dynamic>& perms,
    const Eigen::Ref<const Eigen::Matrix<T, Eigen::Dynamic, 1>>& bc_values1,
    const std::vector<bool>& bc_markers1,
    const Eigen::Ref<const Eigen::Matrix<T, Eigen::Dynamic, 1>>& x0,
    double scale)
{
  const int gdim = mesh.geometry().dim();
  const int tdim = mesh.topology().dim();

  // Prepare cell geometry
  const graph::AdjacencyList<std::int32_t>& x_dofmap = mesh.geometry().dofmap();
  // FIXME: Add proper interface for num coordinate dofs
  const int num_dofs_g = x_dofmap.num_links(0);
  const Eigen::Array<double, Eigen::Dynamic, 3, Eigen::RowMajor>& x_g
      = mesh.geometry().x();

  // Data structures used in bc application
  Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
      coordinate_dofs(num_dofs_g, gdim);
  Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> Ae;
  Eigen::Matrix<T, Eigen::Dynamic, 1> be;

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
    const auto* it = std::find(facets.data(), facets.data() + facets.rows(), f);
    assert(it != (facets.data() + facets.rows()));
    const int local_facet = std::distance(facets.data(), it);

    // Get dof maps for cell
    auto dmap1 = dofmap1.links(cell);

    // Check if bc is applied to cell
    bool has_bc = false;
    for (Eigen::Index j = 0; j < dmap1.size(); ++j)
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

    // Get cell vertex coordinates
    auto x_dofs = x_dofmap.links(cell);
    for (int i = 0; i < num_dofs_g; ++i)
      for (int j = 0; j < gdim; ++j)
        coordinate_dofs(i, j) = x_g(x_dofs[i], j);

    // Size data structure for assembly
    auto dmap0 = dofmap0.links(cell);

    // TODO: Move gathering of coefficients outside of main assembly
    // loop

    auto coeff_array = coeffs.row(cell);
    Ae.setZero(bs0 * dmap0.size(), bs1 * dmap1.size());
    kernel(Ae.data(), coeff_array.data(), constant_values.data(),
           coordinate_dofs.data(), &local_facet, &perms(local_facet, cell),
           cell_info[cell]);

    // Size data structure for assembly
    be.setZero(bs0 * dmap0.size());
    for (Eigen::Index j = 0; j < dmap1.size(); ++j)
    {
      for (int k = 0; k < bs1; ++k)
      {
        const std::int32_t jj = bs1 * dmap1[j] + k;
        if (bc_markers1[jj])
        {
          const T bc = bc_values1[jj];
          if (x0.rows() > 0)
            be -= Ae.col(bs1 * j + k) * scale * (bc - x0[jj]);
          else
            be -= Ae.col(bs1 * j + k) * scale * bc;
        }
      }
    }

    for (Eigen::Index i = 0; i < dmap0.size(); ++i)
      for (int k = 0; k < bs0; ++k)
        b[bs0 * dmap0[i] + k] += be[bs0 * i + k];
  }
}
//----------------------------------------------------------------------------
template <typename T>
void _lift_bc_interior_facets(
    Eigen::Ref<Eigen::Matrix<T, Eigen::Dynamic, 1>> b, const mesh::Mesh& mesh,
    const std::function<void(T*, const T*, const T*, const double*, const int*,
                             const std::uint8_t*, const std::uint32_t)>& kernel,
    const std::vector<std::int32_t>& active_facets,
    const graph::AdjacencyList<std::int32_t>& dofmap0, int bs0,
    const graph::AdjacencyList<std::int32_t>& dofmap1, int bs1,
    const Eigen::Array<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>&
        coeffs,
    const std::vector<int>& offsets,
    const Eigen::Array<T, Eigen::Dynamic, 1>& constant_values,
    const Eigen::Array<std::uint32_t, Eigen::Dynamic, 1>& cell_info,
    const Eigen::Array<std::uint8_t, Eigen::Dynamic, Eigen::Dynamic>& perms,
    const Eigen::Ref<const Eigen::Matrix<T, Eigen::Dynamic, 1>>& bc_values1,
    const std::vector<bool>& bc_markers1,
    const Eigen::Ref<const Eigen::Matrix<T, Eigen::Dynamic, 1>>& x0,
    double scale)
{
  throw std::runtime_error(
      "_lift_bc_interior_facets code looks wrong. Needs checking.");

  const int gdim = mesh.geometry().dim();
  const int tdim = mesh.topology().dim();

  // Prepare cell geometry
  const graph::AdjacencyList<std::int32_t>& x_dofmap = mesh.geometry().dofmap();
  // FIXME: Add proper interface for num coordinate dofs
  const int num_dofs_g = x_dofmap.num_links(0);
  const Eigen::Array<double, Eigen::Dynamic, 3, Eigen::RowMajor>& x_g
      = mesh.geometry().x();

  // Data structures used in assembly
  Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
      coordinate_dofs(2 * num_dofs_g, gdim);
  Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> Ae;
  Eigen::Array<T, Eigen::Dynamic, 1> coeff_array(2 * offsets.back());
  assert(offsets.back() == coeffs.cols());
  Eigen::Matrix<T, Eigen::Dynamic, 1> be;

  // Temporaries for joint dofmaps
  Eigen::Array<std::int32_t, Eigen::Dynamic, 1> dmapjoint0, dmapjoint1;

  const mesh::Topology& topology = mesh.topology();
  auto connectivity = topology.connectivity(tdim - 1, tdim);
  assert(connectivity);
  auto c_to_f = topology.connectivity(tdim, tdim - 1);
  assert(c_to_f);
  auto f_to_c = topology.connectivity(tdim - 1, tdim);
  assert(f_to_c);
  auto map = topology.index_map(tdim - 1);
  assert(map);

  for (std::int32_t f : active_facets)
  {
    // Create attached cells
    auto cells = f_to_c->links(f);
    assert(cells.rows() == 2);

    // Get local index of facet with respect to the cell
    auto facets0 = c_to_f->links(cells[0]);
    const auto* it0
        = std::find(facets0.data(), facets0.data() + facets0.rows(), f);
    assert(it0 != (facets0.data() + facets0.rows()));
    const int local_facet0 = std::distance(facets0.data(), it0);
    auto facets1 = c_to_f->links(cells[1]);
    const auto* it1
        = std::find(facets1.data(), facets1.data() + facets1.rows(), f);
    assert(it1 != (facets1.data() + facets1.rows()));
    const int local_facet1 = std::distance(facets1.data(), it1);

    const std::array local_facet{local_facet0, local_facet1};

    // Get cell geometry
    auto x_dofs0 = x_dofmap.links(cells[0]);
    auto x_dofs1 = x_dofmap.links(cells[1]);
    for (int i = 0; i < num_dofs_g; ++i)
    {
      for (int j = 0; j < gdim; ++j)
      {
        coordinate_dofs(i, j) = x_g(x_dofs0[i], j);
        coordinate_dofs(i + num_dofs_g, j) = x_g(x_dofs1[i], j);
      }
    }

    // Get dof maps for cells and pack
    auto dmap0_cell0 = dofmap0.links(cells[0]);
    auto dmap0_cell1 = dofmap0.links(cells[1]);
    dmapjoint0.resize(dmap0_cell0.size() + dmap0_cell1.size());
    dmapjoint0.head(dmap0_cell0.size()) = dmap0_cell0;
    dmapjoint0.tail(dmap0_cell1.size()) = dmap0_cell1;

    auto dmap1_cell0 = dofmap1.links(cells[0]);
    auto dmap1_cell1 = dofmap1.links(cells[1]);
    dmapjoint1.resize(dmap1_cell0.size() + dmap1_cell1.size());
    dmapjoint1.head(dmap1_cell0.size()) = dmap1_cell0;
    dmapjoint1.tail(dmap1_cell1.size()) = dmap1_cell1;

    // Check if bc is applied to cell
    bool has_bc = false;
    for (Eigen::Index j = 0; j < dmap1_cell0.size(); ++j)
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

    for (Eigen::Index j = 0; j < dmap1_cell1.size(); ++j)
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
    auto coeff_cell0 = coeffs.row(cells[0]);
    auto coeff_cell1 = coeffs.row(cells[1]);

    // Loop over coefficients
    for (std::size_t i = 0; i < offsets.size() - 1; ++i)
    {
      // Loop over entries for coefficient i
      const int num_entries = offsets[i + 1] - offsets[i];
      coeff_array.segment(2 * offsets[i], num_entries)
          = coeff_cell0.segment(offsets[i], num_entries);
      coeff_array.segment(offsets[i + 1] + offsets[i], num_entries)
          = coeff_cell1.segment(offsets[i], num_entries);
    }

    // Tabulate tensor
    Ae.setZero(bs0 * dmapjoint0.size(), bs1 * dmapjoint1.size());
    const std::array perm{perms(local_facet[0], cells[0]),
                          perms(local_facet[1], cells[1])};
    kernel(Ae.data(), coeff_array.data(), constant_values.data(),
           coordinate_dofs.data(), local_facet.data(), perm.data(),
           cell_info[cells[0]]);

    // Compute b = b - A*b for cell0
    be.setZero(bs0 * (dmap0_cell0.size() + dmap0_cell1.size()));
    for (Eigen::Index j = 0; j < dmap1_cell0.size(); ++j)
    {
      for (int k = 0; k < bs1; ++k)
      {
        const std::int32_t jj = bs1 * dmap1_cell0[j] + k;
        if (bc_markers1[jj])
        {
          const T bc = bc_values1[jj];
          if (x0.rows() > 0)
            be -= Ae.col(bs1 * j + k) * scale * (bc - x0[jj]);
          else
            be -= Ae.col(bs1 * j + k) * scale * bc;
        }
      }
    }

    for (Eigen::Index i = 0; i < dmap0_cell0.size(); ++i)
      for (int k = 0; k < bs0; ++k)
        b[bs0 * dmap0_cell0[i] + k] += be[bs0 * i + k];

    // Compute b = b - A*b for cell1
    be.setZero(bs0 * (dmap0_cell0.size() + dmap0_cell1.size()));
    for (Eigen::Index j = 0; j < dmap1_cell1.size(); ++j)
    {
      for (int k = 0; k < bs1; ++k)
      {
        const std::int32_t jj = bs1 * dmap1_cell1[j] + k;
        if (bc_markers1[jj])
        {
          const T bc = bc_values1[jj];
          if (x0.rows() > 0)
            be -= Ae.col(bs1 * j + k) * scale * (bc - x0[jj]);
          else
            be -= Ae.col(bs1 * j + k) * scale * bc;
        }
      }
    }

    for (Eigen::Index i = 0; i < dmap0_cell1.size(); ++i)
      for (int k = 0; k < bs0; ++k)
        b[bs0 * dmap0_cell1[i] + k] += be[bs0 * i + k];
  }
}
//-----------------------------------------------------------------------------
template <typename T>
void assemble_vector(Eigen::Ref<Eigen::Matrix<T, Eigen::Dynamic, 1>> b,
                     const Form<T>& L)
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
  const Eigen::Array<T, Eigen::Dynamic, 1> constant_values = pack_constants(L);

  // Prepare coefficients
  const Eigen::Array<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> coeffs
      = pack_coefficients(L);

  const bool needs_permutation_data = L.needs_permutation_data();
  if (needs_permutation_data)
    mesh->topology_mutable().create_entity_permutations();
  const Eigen::Array<std::uint32_t, Eigen::Dynamic, 1>& cell_info
      = needs_permutation_data
            ? mesh->topology().get_cell_permutation_info()
            : Eigen::Array<std::uint32_t, Eigen::Dynamic, 1>(num_cells);

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

    const int facets_per_cell
        = mesh::cell_num_entities(mesh->topology().cell_type(), tdim - 1);
    const Eigen::Array<std::uint8_t, Eigen::Dynamic, Eigen::Dynamic>& perms
        = needs_permutation_data
              ? mesh->topology().get_facet_permutations()
              : Eigen::Array<std::uint8_t, Eigen::Dynamic, Eigen::Dynamic>(
                  facets_per_cell, num_cells);
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
    Eigen::Ref<Eigen::Matrix<T, Eigen::Dynamic, 1>> b,
    const mesh::Geometry& geometry,
    const std::vector<std::int32_t>& active_cells,
    const graph::AdjacencyList<std::int32_t>& dofmap, const int bs,
    const std::function<void(T*, const T*, const T*, const double*, const int*,
                             const std::uint8_t*, const std::uint32_t)>& kernel,
    const Eigen::Array<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>&
        coeffs,
    const Eigen::Array<T, Eigen::Dynamic, 1>& constant_values,
    const Eigen::Array<std::uint32_t, Eigen::Dynamic, 1>& cell_info)
{
  const int gdim = geometry.dim();

  // Prepare cell geometry
  const graph::AdjacencyList<std::int32_t>& x_dofmap = geometry.dofmap();

  // FIXME: Add proper interface for num coordinate dofs
  const int num_dofs_g = x_dofmap.num_links(0);
  const Eigen::Array<double, Eigen::Dynamic, 3, Eigen::RowMajor>& x_g
      = geometry.x();

  // FIXME: Add proper interface for num_dofs
  // Create data structures used in assembly
  const int num_dofs = dofmap.links(0).size();
  Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
      coordinate_dofs(num_dofs_g, gdim);
  Eigen::Matrix<T, Eigen::Dynamic, 1> be(bs * num_dofs);

  // Iterate over active cells
  for (std::int32_t c : active_cells)
  {
    // Get cell coordinates/geometry
    auto x_dofs = x_dofmap.links(c);
    for (int i = 0; i < num_dofs_g; ++i)
      for (int j = 0; j < gdim; ++j)
        coordinate_dofs(i, j) = x_g(x_dofs[i], j);

    // Tabulate vector for cell
    std::fill(be.data(), be.data() + bs * num_dofs, 0);
    kernel(be.data(), coeffs.row(c).data(), constant_values.data(),
           coordinate_dofs.data(), nullptr, nullptr, cell_info[c]);

    // Scatter cell vector to 'global' vector array
    auto dofs = dofmap.links(c);
    for (Eigen::Index i = 0; i < num_dofs; ++i)
      for (int k = 0; k < bs; ++k)
        b[bs * dofs[i] + k] += be[bs * i + k];
  }
}
//-----------------------------------------------------------------------------
template <typename T>
void assemble_exterior_facets(
    Eigen::Ref<Eigen::Matrix<T, Eigen::Dynamic, 1>> b, const mesh::Mesh& mesh,
    const std::vector<std::int32_t>& active_facets,
    const graph::AdjacencyList<std::int32_t>& dofmap, const int bs,
    const std::function<void(T*, const T*, const T*, const double*, const int*,
                             const std::uint8_t*, const std::uint32_t)>& fn,
    const Eigen::Array<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>&
        coeffs,
    const Eigen::Array<T, Eigen::Dynamic, 1>& constant_values,
    const Eigen::Array<std::uint32_t, Eigen::Dynamic, 1>& cell_info,
    const Eigen::Array<std::uint8_t, Eigen::Dynamic, Eigen::Dynamic>& perms)
{
  const int gdim = mesh.geometry().dim();
  const int tdim = mesh.topology().dim();

  // Prepare cell geometry
  const graph::AdjacencyList<std::int32_t>& x_dofmap = mesh.geometry().dofmap();

  // FIXME: Add proper interface for num coordinate dofs
  const int num_dofs_g = x_dofmap.num_links(0);
  const Eigen::Array<double, Eigen::Dynamic, 3, Eigen::RowMajor>& x_g
      = mesh.geometry().x();

  // FIXME: Add proper interface for num_dofs
  // Create data structures used in assembly
  const int num_dofs = dofmap.links(0).size();
  Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
      coordinate_dofs(num_dofs_g, gdim);
  Eigen::Matrix<T, Eigen::Dynamic, 1> be(bs * num_dofs);

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
    const auto* it = std::find(facets.data(), facets.data() + facets.rows(), f);
    assert(it != (facets.data() + facets.rows()));
    const int local_facet = std::distance(facets.data(), it);

    // Get cell coordinates/geometry
    auto x_dofs = x_dofmap.links(cell);
    for (int i = 0; i < num_dofs_g; ++i)
      for (int j = 0; j < gdim; ++j)
        coordinate_dofs(i, j) = x_g(x_dofs[i], j);

    // Tabulate element vector
    std::fill(be.data(), be.data() + bs * num_dofs, 0);
    fn(be.data(), coeffs.row(cell).data(), constant_values.data(),
       coordinate_dofs.data(), &local_facet, &perms(local_facet, cell),
       cell_info[cell]);

    // Add element vector to global vector
    auto dofs = dofmap.links(cell);
    for (Eigen::Index i = 0; i < num_dofs; ++i)
      for (int k = 0; k < bs; ++k)
        b[bs * dofs[i] + k] += be[bs * i + k];
  }
}
//-----------------------------------------------------------------------------
template <typename T>
void assemble_interior_facets(
    Eigen::Ref<Eigen::Matrix<T, Eigen::Dynamic, 1>> b, const mesh::Mesh& mesh,
    const std::vector<std::int32_t>& active_facets, const fem::DofMap& dofmap,
    const std::function<void(T*, const T*, const T*, const double*, const int*,
                             const std::uint8_t*, const std::uint32_t)>& fn,
    const Eigen::Array<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>&
        coeffs,
    const std::vector<int>& offsets,
    const Eigen::Array<T, Eigen::Dynamic, 1>& constant_values,
    const Eigen::Array<std::uint32_t, Eigen::Dynamic, 1>& cell_info,
    const Eigen::Array<std::uint8_t, Eigen::Dynamic, Eigen::Dynamic>& perms)
{
  const int gdim = mesh.geometry().dim();
  const int tdim = mesh.topology().dim();

  // Prepare cell geometry
  const graph::AdjacencyList<std::int32_t>& x_dofmap = mesh.geometry().dofmap();
  // FIXME: Add proper interface for num coordinate dofs

  const int num_dofs_g = x_dofmap.num_links(0);
  const Eigen::Array<double, Eigen::Dynamic, 3, Eigen::RowMajor>& x_g
      = mesh.geometry().x();

  // Creat data structures used in assembly
  Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
      coordinate_dofs(2 * num_dofs_g, gdim);
  Eigen::Matrix<T, Eigen::Dynamic, 1> be;
  Eigen::Array<T, Eigen::Dynamic, 1> coeff_array(2 * offsets.back());
  assert(offsets.back() == coeffs.cols());

  auto f_to_c = mesh.topology().connectivity(tdim - 1, tdim);
  assert(f_to_c);
  auto c_to_f = mesh.topology().connectivity(tdim, tdim - 1);
  assert(c_to_f);
  for (const auto& f : active_facets)
  {
    // Get attached cell indices
    auto cells = f_to_c->links(f);
    assert(cells.rows() == 2);

    // Create attached cells
    std::array<int, 2> local_facet;
    for (int i = 0; i < 2; ++i)
    {
      auto facets = c_to_f->links(cells[i]);
      const auto* it
          = std::find(facets.data(), facets.data() + facets.rows(), f);
      assert(it != (facets.data() + facets.rows()));
      local_facet[i] = std::distance(facets.data(), it);
    }

    // Get cell geometry
    auto x_dofs0 = x_dofmap.links(cells[0]);
    auto x_dofs1 = x_dofmap.links(cells[1]);
    for (int i = 0; i < num_dofs_g; ++i)
    {
      for (int j = 0; j < gdim; ++j)
      {
        coordinate_dofs(i, j) = x_g(x_dofs0[i], j);
        coordinate_dofs(i + num_dofs_g, j) = x_g(x_dofs1[i], j);
      }
    }

    // Get dofmaps for cell
    auto dmap0 = dofmap.cell_dofs(cells[0]);
    auto dmap1 = dofmap.cell_dofs(cells[1]);

    // Layout for the restricted coefficients is flattened
    // w[coefficient][restriction][dof]
    auto coeff_cell0 = coeffs.row(cells[0]);
    auto coeff_cell1 = coeffs.row(cells[1]);

    // Loop over coefficients
    for (std::size_t i = 0; i < offsets.size() - 1; ++i)
    {
      // Loop over entries for coefficient i
      const int num_entries = offsets[i + 1] - offsets[i];
      coeff_array.segment(2 * offsets[i], num_entries)
          = coeff_cell0.segment(offsets[i], num_entries);
      coeff_array.segment(offsets[i + 1] + offsets[i], num_entries)
          = coeff_cell1.segment(offsets[i], num_entries);
    }

    // Tabulate element vector
    be.setZero(dmap0.size() + dmap1.size());

    const std::array perm{perms(local_facet[0], cells[0]),
                          perms(local_facet[1], cells[1])};
    fn(be.data(), coeff_array.data(), constant_values.data(),
       coordinate_dofs.data(), local_facet.data(), perm.data(),
       cell_info[cells[0]]);

    // Add element vector to global vector
    for (Eigen::Index i = 0; i < dmap0.size(); ++i)
      b[dmap0[i]] += be[i];
    for (Eigen::Index i = 0; i < dmap1.size(); ++i)
      b[dmap1[i]] += be[i + dmap0.size()];
  }
}
//-----------------------------------------------------------------------------
template <typename T>
void apply_lifting(
    Eigen::Ref<Eigen::Matrix<T, Eigen::Dynamic, 1>> b,
    const std::vector<std::shared_ptr<const Form<T>>> a,
    const std::vector<std::vector<std::shared_ptr<const DirichletBC<T>>>>& bcs1,
    const std::vector<Eigen::Ref<const Eigen::Matrix<T, Eigen::Dynamic, 1>>>&
        x0,
    double scale)
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
    Eigen::Matrix<T, Eigen::Dynamic, 1> bc_values1;
    if (a[j] and !bcs1[j].empty())
    {
      auto V1 = a[j]->function_spaces()[1];
      assert(V1);
      auto map1 = V1->dofmap()->index_map;
      const int bs1 = V1->dofmap()->index_map_bs();
      assert(map1);
      const int crange = bs1 * (map1->size_local() + map1->num_ghosts());
      bc_markers1.assign(crange, false);
      bc_values1 = Eigen::Matrix<T, Eigen::Dynamic, 1>::Zero(crange);
      for (const std::shared_ptr<const DirichletBC<T>>& bc : bcs1[j])
      {
        bc->mark_dofs(bc_markers1);
        bc->dof_values(bc_values1);
      }

      if (!x0.empty())
        lift_bc<T>(b, *a[j], bc_values1, bc_markers1, x0[j], scale);
      else
      {
        const Eigen::Matrix<T, Eigen::Dynamic, 1> x0(0);
        lift_bc<T>(b, *a[j], bc_values1, bc_markers1, x0, scale);
      }
    }
  }
}
//-----------------------------------------------------------------------------
template <typename T>
void lift_bc(
    Eigen::Ref<Eigen::Matrix<T, Eigen::Dynamic, 1>> b, const Form<T>& a,
    const Eigen::Ref<const Eigen::Matrix<T, Eigen::Dynamic, 1>>& bc_values1,
    const std::vector<bool>& bc_markers1,
    const Eigen::Ref<const Eigen::Matrix<T, Eigen::Dynamic, 1>>& x0,
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
  const int bs1 = a.function_spaces()[0]->dofmap()->bs();

  // Prepare constants
  const Eigen::Array<T, Eigen::Dynamic, 1> constant_values = pack_constants(a);

  // Prepare coefficients
  const Eigen::Array<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> coeffs
      = pack_coefficients(a);

  const bool needs_permutation_data = a.needs_permutation_data();
  if (needs_permutation_data)
    mesh->topology_mutable().create_entity_permutations();
  const Eigen::Array<std::uint32_t, Eigen::Dynamic, 1>& cell_info
      = needs_permutation_data
            ? mesh->topology().get_cell_permutation_info()
            : Eigen::Array<std::uint32_t, Eigen::Dynamic, 1>(num_cells);

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

    const int facets_per_cell
        = mesh::cell_num_entities(mesh->topology().cell_type(), tdim - 1);
    const Eigen::Array<std::uint8_t, Eigen::Dynamic, Eigen::Dynamic>& perms
        = needs_permutation_data
              ? mesh->topology().get_facet_permutations()
              : Eigen::Array<std::uint8_t, Eigen::Dynamic, Eigen::Dynamic>(
                  facets_per_cell, num_cells);
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
