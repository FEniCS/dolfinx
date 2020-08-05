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
    const graph::AdjacencyList<std::int32_t>& dofmap,
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
    const graph::AdjacencyList<std::int32_t>& dofmap,
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

/// Modify RHS vector to account for boundary condition
///
///    b <- b - scale * A x_bc
////
/// @param[in,out] b The vector to be modified
/// @param[in] a The bilinear form that generates A
/// @param[in] bc_values1 The boundary condition 'values'
/// @param[in] bc_markers1 The indices (columns of A, rows of x) to
///                        which bcs belong
/// @param[in] scale Scaling to apply
template <typename T>
void lift_bc(
    Eigen::Ref<Eigen::Matrix<T, Eigen::Dynamic, 1>> b, const Form<T>& a,
    const Eigen::Ref<const Eigen::Matrix<T, Eigen::Dynamic, 1>>& bc_values1,
    const std::vector<bool>& bc_markers1, double scale);

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
    Eigen::Ref<Eigen::Matrix<T, Eigen::Dynamic, 1>> b, const Form<T>& a,
    const Eigen::Ref<const Eigen::Matrix<T, Eigen::Dynamic, 1>>& bc_values1,
    const std::vector<bool>& bc_markers1,
    const Eigen::Ref<const Eigen::Matrix<T, Eigen::Dynamic, 1>>& x0,
    double scale)
{
  assert(a.rank() == 2);

  // Get mesh from form
  std::shared_ptr<const mesh::Mesh> mesh = a.mesh();
  assert(mesh);

  mesh->topology_mutable().create_entity_permutations();

  // Get dofmap for columns and rows of a
  assert(a.function_space(0));
  assert(a.function_space(1));
  std::shared_ptr<const fem::DofMap> dofmap0 = a.function_space(0)->dofmap();
  std::shared_ptr<const fem::DofMap> dofmap1 = a.function_space(1)->dofmap();
  assert(dofmap0);
  assert(dofmap1);

  // Prepare coefficients
  Eigen::Array<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> coeffs
      = pack_coefficients(a);

  const std::function<void(T*, const T*, const T*, const double*, const int*,
                           const std::uint8_t*, const std::uint32_t)>& fn
      = a.integrals().get_tabulate_tensor(IntegralType::cell, 0);

  // Prepare cell geometry
  const int gdim = mesh->geometry().dim();
  const graph::AdjacencyList<std::int32_t>& x_dofmap
      = mesh->geometry().dofmap();

  // FIXME: Add proper interface for num coordinate dofs
  const int num_dofs_g = x_dofmap.num_links(0);
  const Eigen::Array<double, Eigen::Dynamic, 3, Eigen::RowMajor>& x_g
      = mesh->geometry().x();

  // Data structures used in bc application
  Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
      coordinate_dofs(num_dofs_g, gdim);
  Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> Ae;
  Eigen::Matrix<T, Eigen::Dynamic, 1> be;

  // Prepare constants
  if (!a.all_constants_set())
    throw std::runtime_error("Unset constant in Form");
  const Eigen::Array<T, Eigen::Dynamic, 1> constant_values = pack_constants(a);

  const Eigen::Array<std::uint32_t, Eigen::Dynamic, 1>& cell_info
      = mesh->topology().get_cell_permutation_info();

  // Iterate over all cells
  const int tdim = mesh->topology().dim();
  auto map = mesh->topology().index_map(tdim);
  assert(map);
  const int num_cells = map->size_local();
  for (int c = 0; c < num_cells; ++c)
  {
    // Get dof maps for cell
    auto dmap1 = dofmap1->cell_dofs(c);

    // Check if bc is applied to cell
    bool has_bc = false;
    for (Eigen::Index j = 0; j < dmap1.size(); ++j)
    {
      if (bc_markers1[dmap1[j]])
      {
        has_bc = true;
        break;
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
    auto dmap0 = dofmap0->cell_dofs(c);

    auto coeff_array = coeffs.row(c);
    Ae.setZero(dmap0.size(), dmap1.size());
    fn(Ae.data(), coeff_array.data(), constant_values.data(),
       coordinate_dofs.data(), nullptr, nullptr, cell_info[c]);

    // Size data structure for assembly
    be.setZero(dmap0.size());
    for (Eigen::Index j = 0; j < dmap1.size(); ++j)
    {
      const std::int32_t jj = dmap1[j];
      if (bc_markers1[jj])
      {
        const T bc = bc_values1[jj];
        if (x0.rows() > 0)
          be -= Ae.col(j) * scale * (bc - x0[jj]);
        else
          be -= Ae.col(j) * scale * bc;
      }
    }

    for (Eigen::Index k = 0; k < dmap0.size(); ++k)
      b[dmap0[k]] += be[k];
  }
}
//----------------------------------------------------------------------------
template <typename T>
void _lift_bc_exterior_facets(
    Eigen::Ref<Eigen::Matrix<T, Eigen::Dynamic, 1>> b, const Form<T>& a,
    const Eigen::Ref<const Eigen::Matrix<T, Eigen::Dynamic, 1>>& bc_values1,
    const std::vector<bool>& bc_markers1,
    const Eigen::Ref<const Eigen::Matrix<T, Eigen::Dynamic, 1>>& x0,
    double scale)
{
  assert(a.rank() == 2);

  // Get mesh from form
  std::shared_ptr<const mesh::Mesh> mesh = a.mesh();
  assert(mesh);

  mesh->topology_mutable().create_entity_permutations();

  const int gdim = mesh->geometry().dim();
  const int tdim = mesh->topology().dim();

  // FIXME: cleanup these calls? Some of the happen internally again.
  mesh->topology_mutable().create_entities(tdim - 1);
  mesh->topology_mutable().create_connectivity(tdim - 1, tdim);
  // FIXME: Why again -- appears already See 8 lines above.
  mesh->topology_mutable().create_entity_permutations();

  // Get dofmap for columns and rows of a
  assert(a.function_space(0));
  assert(a.function_space(1));
  std::shared_ptr<const fem::DofMap> dofmap0 = a.function_space(0)->dofmap();
  std::shared_ptr<const fem::DofMap> dofmap1 = a.function_space(1)->dofmap();
  assert(dofmap0);
  assert(dofmap1);

  // Prepare coefficients
  Eigen::Array<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> coeffs
      = pack_coefficients(a);

  const std::function<void(T*, const T*, const T*, const double*, const int*,
                           const std::uint8_t*, const std::uint32_t)>& fn
      = a.integrals().get_tabulate_tensor(IntegralType::exterior_facet, 0);

  // Prepare cell geometry
  const graph::AdjacencyList<std::int32_t>& x_dofmap
      = mesh->geometry().dofmap();
  // FIXME: Add proper interface for num coordinate dofs
  const int num_dofs_g = x_dofmap.num_links(0);
  const Eigen::Array<double, Eigen::Dynamic, 3, Eigen::RowMajor>& x_g
      = mesh->geometry().x();

  // Data structures used in bc application
  Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
      coordinate_dofs(num_dofs_g, gdim);
  Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> Ae;
  Eigen::Matrix<T, Eigen::Dynamic, 1> be;

  // Prepare constants
  if (!a.all_constants_set())
    throw std::runtime_error("Unset constant in Form");
  const Eigen::Array<T, Eigen::Dynamic, 1> constant_values = pack_constants(a);

  // Iterate over owned facets
  const mesh::Topology& topology = mesh->topology();
  auto connectivity = topology.connectivity(tdim - 1, tdim);
  assert(connectivity);
  auto c_to_f = topology.connectivity(tdim, tdim - 1);
  assert(c_to_f);
  auto map = topology.index_map(tdim - 1);
  assert(map);

  const Eigen::Array<std::uint8_t, Eigen::Dynamic, Eigen::Dynamic>& perms
      = mesh->topology().get_facet_permutations();
  const Eigen::Array<std::uint32_t, Eigen::Dynamic, 1>& cell_info
      = mesh->topology().get_cell_permutation_info();

  std::set<std::int32_t> fwd_shared_facets;
  // Only need to consider shared facets when there are no ghost cells
  if (topology.index_map(tdim)->num_ghosts() == 0)
  {
    fwd_shared_facets.insert(
        topology.index_map(tdim - 1)->shared_indices().begin(),
        topology.index_map(tdim - 1)->shared_indices().end());
  }

  for (int f = 0; f < map->size_local(); ++f)
  {
    // Move to next facet if this one is an interior facet
    // Interior facets have two attached cells. If on a process boundary,
    // and owned locally, then they are "forward shared".
    if (connectivity->num_links(f) == 2
        or fwd_shared_facets.find(f) != fwd_shared_facets.end())
      continue;

    // Create attached cell
    assert(connectivity->num_links(f) == 1);
    const std::int32_t cell = connectivity->links(f)[0];

    // Get local index of facet with respect to the cell
    auto facets = c_to_f->links(cell);
    const auto* it = std::find(facets.data(), facets.data() + facets.rows(), f);
    assert(it != (facets.data() + facets.rows()));
    const int local_facet = std::distance(facets.data(), it);

    const std::uint8_t perm = perms(local_facet, cell);

    // Get dof maps for cell
    auto dmap1 = dofmap1->cell_dofs(cell);

    // Check if bc is applied to cell
    bool has_bc = false;
    for (Eigen::Index j = 0; j < dmap1.size(); ++j)
    {
      if (bc_markers1[dmap1[j]])
      {
        has_bc = true;
        break;
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
    auto dmap0 = dofmap0->cell_dofs(cell);

    // TODO: Move gathering of coefficients outside of main assembly
    // loop

    auto coeff_array = coeffs.row(cell);
    Ae.setZero(dmap0.size(), dmap1.size());
    fn(Ae.data(), coeff_array.data(), constant_values.data(),
       coordinate_dofs.data(), &local_facet, &perm, cell_info[cell]);

    // Size data structure for assembly
    be.setZero(dmap0.size());
    for (Eigen::Index j = 0; j < dmap1.size(); ++j)
    {
      const std::int32_t jj = dmap1[j];
      if (bc_markers1[jj])
      {
        const T bc = bc_values1[jj];
        if (x0.rows() > 0)
          be -= Ae.col(j) * scale * (bc - x0[jj]);
        else
          be -= Ae.col(j) * scale * bc;
      }
    }

    for (Eigen::Index k = 0; k < dmap0.size(); ++k)
      b[dmap0[k]] += be[k];
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
  assert(L.function_space(0));
  std::shared_ptr<const fem::DofMap> dofmap = L.function_space(0)->dofmap();
  assert(dofmap);
  const graph::AdjacencyList<std::int32_t>& dofs = dofmap->list();

  // Prepare constants
  if (!L.all_constants_set())
    throw std::runtime_error("Unset constant in Form");
  const Eigen::Array<T, Eigen::Dynamic, 1> constant_values = pack_constants(L);

  // Prepare coefficients
  const Eigen::Array<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> coeffs
      = pack_coefficients(L);

  const FormIntegrals<T>& integrals = L.integrals();
  const bool needs_permutation_data = integrals.needs_permutation_data();
  if (needs_permutation_data)
    mesh->topology_mutable().create_entity_permutations();
  const Eigen::Array<std::uint32_t, Eigen::Dynamic, 1>& cell_info
      = needs_permutation_data
            ? mesh->topology().get_cell_permutation_info()
            : Eigen::Array<std::uint32_t, Eigen::Dynamic, 1>(num_cells);

  for (int i = 0; i < integrals.num_integrals(IntegralType::cell); ++i)
  {
    const auto& fn = integrals.get_tabulate_tensor(IntegralType::cell, i);
    const std::vector<std::int32_t>& active_cells
        = integrals.integral_domains(IntegralType::cell, i);
    fem::impl::assemble_cells(b, mesh->geometry(), active_cells, dofs, fn,
                              coeffs, constant_values, cell_info);
  }

  if (integrals.num_integrals(IntegralType::exterior_facet) > 0
      or integrals.num_integrals(IntegralType::interior_facet) > 0)
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
    for (int i = 0; i < integrals.num_integrals(IntegralType::exterior_facet);
         ++i)
    {
      const auto& fn
          = integrals.get_tabulate_tensor(IntegralType::exterior_facet, i);
      const std::vector<std::int32_t>& active_facets
          = integrals.integral_domains(IntegralType::exterior_facet, i);
      fem::impl::assemble_exterior_facets(b, *mesh, active_facets, dofs, fn,
                                          coeffs, constant_values, cell_info,
                                          perms);
    }

    const std::vector<int> c_offsets = L.coefficients().offsets();
    for (int i = 0; i < integrals.num_integrals(IntegralType::interior_facet);
         ++i)
    {
      const auto& fn
          = integrals.get_tabulate_tensor(IntegralType::interior_facet, i);
      const std::vector<std::int32_t>& active_facets
          = integrals.integral_domains(IntegralType::interior_facet, i);
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
    const graph::AdjacencyList<std::int32_t>& dofmap,
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
  Eigen::Matrix<T, Eigen::Dynamic, 1> be(num_dofs);

  // Iterate over active cells
  for (std::int32_t c : active_cells)
  {
    // Get cell coordinates/geometry
    auto x_dofs = x_dofmap.links(c);
    for (int i = 0; i < num_dofs_g; ++i)
      for (int j = 0; j < gdim; ++j)
        coordinate_dofs(i, j) = x_g(x_dofs[i], j);

    // Tabulate vector for cell
    std::fill(be.data(), be.data() + num_dofs, 0);
    kernel(be.data(), coeffs.row(c).data(), constant_values.data(),
           coordinate_dofs.data(), nullptr, nullptr, cell_info[c]);

    // Scatter cell vector to 'global' vector array
    auto dofs = dofmap.links(c);
    for (Eigen::Index i = 0; i < num_dofs; ++i)
      b[dofs[i]] += be[i];
  }
}
//-----------------------------------------------------------------------------
template <typename T>
void assemble_exterior_facets(
    Eigen::Ref<Eigen::Matrix<T, Eigen::Dynamic, 1>> b, const mesh::Mesh& mesh,
    const std::vector<std::int32_t>& active_facets,
    const graph::AdjacencyList<std::int32_t>& dofmap,
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
  Eigen::Matrix<T, Eigen::Dynamic, 1> be(num_dofs);

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
    std::fill(be.data(), be.data() + num_dofs, 0);
    fn(be.data(), coeffs.row(cell).data(), constant_values.data(),
       coordinate_dofs.data(), &local_facet, &perms(local_facet, cell),
       cell_info[cell]);

    // Add element vector to global vector
    auto dofs = dofmap.links(cell);
    for (Eigen::Index i = 0; i < num_dofs; ++i)
      b[dofs[i]] += be[i];
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
      auto V1 = a[j]->function_space(1);
      assert(V1);
      auto map1 = V1->dofmap()->index_map;
      assert(map1);
      const int crange
          = map1->block_size() * (map1->size_local() + map1->num_ghosts());
      bc_markers1.assign(crange, false);
      bc_values1 = Eigen::Matrix<T, Eigen::Dynamic, 1>::Zero(crange);
      for (const std::shared_ptr<const DirichletBC<T>>& bc : bcs1[j])
      {
        bc->mark_dofs(bc_markers1);
        bc->dof_values(bc_values1);
      }

      // Modify (apply lifting) vector
      if (!x0.empty())
        lift_bc<T>(b, *a[j], bc_values1, bc_markers1, x0[j], scale);
      else
        lift_bc<T>(b, *a[j], bc_values1, bc_markers1, scale);
    }
  }
}
//-----------------------------------------------------------------------------
template <typename T>
void lift_bc(
    Eigen::Ref<Eigen::Matrix<T, Eigen::Dynamic, 1>> b, const Form<T>& a,
    const Eigen::Ref<const Eigen::Matrix<T, Eigen::Dynamic, 1>>& bc_values1,
    const std::vector<bool>& bc_markers1, double scale)
{
  const Eigen::Matrix<T, Eigen::Dynamic, 1> x0(0);
  if (a.integrals().num_integrals(fem::IntegralType::cell) > 0)
    _lift_bc_cells<T>(b, a, bc_values1, bc_markers1, x0, scale);
  if (a.integrals().num_integrals(fem::IntegralType::exterior_facet) > 0)
    _lift_bc_exterior_facets<T>(b, a, bc_values1, bc_markers1, x0, scale);
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
  if (a.integrals().num_integrals(fem::IntegralType::cell) > 0)
    _lift_bc_cells(b, a, bc_values1, bc_markers1, x0, scale);
  if (a.integrals().num_integrals(fem::IntegralType::exterior_facet) > 0)
    _lift_bc_exterior_facets(b, a, bc_values1, bc_markers1, x0, scale);
}
//-----------------------------------------------------------------------------
} // namespace dolfinx::fem::impl
