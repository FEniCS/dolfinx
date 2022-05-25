// Copyright (C) 2018-2019 Garth N. Wells
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include "DofMap.h"
#include "Form.h"
#include "FunctionSpace.h"
#include "utils.h"
#include <dolfinx/common/utils.h>
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
template <typename T, typename U>
void assemble_matrix(
    U mat_set_values, const Form<T>& a, const xtl::span<const T>& constants,
    const std::map<std::pair<IntegralType, int>,
                   std::pair<xtl::span<const T>, int>>& coefficients,
    const xtl::span<const std::int8_t>& bc0,
    const xtl::span<const std::int8_t>& bc1);

/// Execute kernel over cells and accumulate result in matrix
template <typename T, typename U>
void assemble_cells(
    U mat_set, const mesh::Geometry& geometry,
    const xtl::span<const std::int32_t>& cells,
    const std::function<void(const xtl::span<T>&,
                             const xtl::span<const std::uint32_t>&,
                             std::int32_t, int)>& dof_transform,
    const graph::AdjacencyList<std::int32_t>& dofmap0, int bs0,
    const std::function<void(const xtl::span<T>&,
                             const xtl::span<const std::uint32_t>&,
                             std::int32_t, int)>& dof_transform_to_transpose,
    const graph::AdjacencyList<std::int32_t>& dofmap1, int bs1,
    const xtl::span<const std::int8_t>& bc0,
    const xtl::span<const std::int8_t>& bc1,
    const std::function<void(T*, const T*, const T*, const double*, const int*,
                             const std::uint8_t*)>& kernel,
    const xtl::span<const T>& coeffs, int cstride,
    const xtl::span<const T>& constants,
    const xtl::span<const std::uint32_t>& cell_info)
{
  if (cells.empty())
    return;

  // Prepare cell geometry
  const graph::AdjacencyList<std::int32_t>& x_dofmap = geometry.dofmap();
  const std::size_t num_dofs_g = geometry.cmap().dim();

  xtl::span<const double> x_g = geometry.x();

  // Iterate over active cells
  const int num_dofs0 = dofmap0.links(0).size();
  const int num_dofs1 = dofmap1.links(0).size();
  const int ndim0 = bs0 * num_dofs0;
  const int ndim1 = bs1 * num_dofs1;
  std::vector<T> Ae(ndim0 * ndim1);
  const xtl::span<T> _Ae(Ae);
  std::vector<double> coordinate_dofs(3 * num_dofs_g);

  // Iterate over active cells
  for (std::size_t index = 0; index < cells.size(); ++index)
  {
    std::int32_t c = cells[index];

    // Get cell coordinates/geometry
    auto x_dofs = x_dofmap.links(c);
    for (std::size_t i = 0; i < x_dofs.size(); ++i)
    {
      common::impl::copy_N<3>(std::next(x_g.begin(), 3 * x_dofs[i]),
                              std::next(coordinate_dofs.begin(), 3 * i));
    }

    // Tabulate tensor
    std::fill(Ae.begin(), Ae.end(), 0);
    kernel(Ae.data(), coeffs.data() + index * cstride, constants.data(),
           coordinate_dofs.data(), nullptr, nullptr);

    dof_transform(_Ae, cell_info, c, ndim1);
    dof_transform_to_transpose(_Ae, cell_info, c, ndim0);

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

    mat_set(dofs0, dofs1, Ae);
  }
}

/// Execute kernel over exterior facets and  accumulate result in Mat
template <typename T, typename U>
void assemble_exterior_facets(
    U mat_set, const mesh::Mesh& mesh,
    const xtl::span<const std::pair<std::int32_t, int>>& facets,
    const std::function<void(const xtl::span<T>&,
                             const xtl::span<const std::uint32_t>&,
                             std::int32_t, int)>& dof_transform,
    const graph::AdjacencyList<std::int32_t>& dofmap0, int bs0,
    const std::function<void(const xtl::span<T>&,
                             const xtl::span<const std::uint32_t>&,
                             std::int32_t, int)>& dof_transform_to_transpose,
    const graph::AdjacencyList<std::int32_t>& dofmap1, int bs1,
    const xtl::span<const std::int8_t>& bc0,
    const xtl::span<const std::int8_t>& bc1,
    const std::function<void(T*, const T*, const T*, const double*, const int*,
                             const std::uint8_t*)>& kernel,
    const xtl::span<const T>& coeffs, int cstride,
    const xtl::span<const T>& constants,
    const xtl::span<const std::uint32_t>& cell_info)
{
  if (facets.empty())
    return;

  // Prepare cell geometry
  const graph::AdjacencyList<std::int32_t>& x_dofmap = mesh.geometry().dofmap();
  const std::size_t num_dofs_g = mesh.geometry().cmap().dim();
  xtl::span<const double> x_g = mesh.geometry().x();

  // Data structures used in assembly
  std::vector<double> coordinate_dofs(3 * num_dofs_g);
  const int num_dofs0 = dofmap0.links(0).size();
  const int num_dofs1 = dofmap1.links(0).size();
  const int ndim0 = bs0 * num_dofs0;
  const int ndim1 = bs1 * num_dofs1;
  std::vector<T> Ae(ndim0 * ndim1);
  const xtl::span<T> _Ae(Ae);

  for (std::size_t index = 0; index < facets.size(); ++index)
  {
    std::int32_t cell = facets[index].first;
    int local_facet = facets[index].second;

    // Get cell coordinates/geometry
    auto x_dofs = x_dofmap.links(cell);
    for (std::size_t i = 0; i < x_dofs.size(); ++i)
    {
      common::impl::copy_N<3>(std::next(x_g.begin(), 3 * x_dofs[i]),
                              std::next(coordinate_dofs.begin(), 3 * i));
    }

    // Tabulate tensor
    std::fill(Ae.begin(), Ae.end(), 0);
    kernel(Ae.data(), coeffs.data() + index * cstride, constants.data(),
           coordinate_dofs.data(), &local_facet, nullptr);

    dof_transform(_Ae, cell_info, cell, ndim1);
    dof_transform_to_transpose(_Ae, cell_info, cell, ndim0);

    // Zero rows/columns for essential bcs
    auto dofs0 = dofmap0.links(cell);
    auto dofs1 = dofmap1.links(cell);
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

    mat_set(dofs0, dofs1, Ae);
  }
}

/// Execute kernel over interior facets and  accumulate result in Mat
template <typename T, typename U>
void assemble_interior_facets(
    U mat_set, const mesh::Mesh& mesh,
    const xtl::span<const std::tuple<std::int32_t, int, std::int32_t, int>>&
        facets,
    const std::function<void(const xtl::span<T>&,
                             const xtl::span<const std::uint32_t>&,
                             std::int32_t, int)>& dof_transform,
    const DofMap& dofmap0, int bs0,
    const std::function<void(const xtl::span<T>&,
                             const xtl::span<const std::uint32_t>&,
                             std::int32_t, int)>& dof_transform_to_transpose,
    const DofMap& dofmap1, int bs1, const xtl::span<const std::int8_t>& bc0,
    const xtl::span<const std::int8_t>& bc1,
    const std::function<void(T*, const T*, const T*, const double*, const int*,
                             const std::uint8_t*)>& kernel,
    const xtl::span<const T>& coeffs, int cstride,
    const xtl::span<const int>& offsets, const xtl::span<const T>& constants,
    const xtl::span<const std::uint32_t>& cell_info,
    const std::function<std::uint8_t(std::size_t)>& get_perm)
{
  if (facets.empty())
    return;

  const int tdim = mesh.topology().dim();

  // Prepare cell geometry
  const graph::AdjacencyList<std::int32_t>& x_dofmap = mesh.geometry().dofmap();

  const std::size_t num_dofs_g = mesh.geometry().cmap().dim();

  xtl::span<const double> x_g = mesh.geometry().x();

  // Data structures used in assembly
  xt::xtensor<double, 3> coordinate_dofs({2, num_dofs_g, 3});
  std::vector<T> Ae, be;
  std::vector<T> coeff_array(2 * offsets.back());
  assert(offsets.back() == cstride);

  const int num_cell_facets
      = mesh::cell_num_entities(mesh.topology().cell_type(), tdim - 1);

  // Temporaries for joint dofmaps
  std::vector<std::int32_t> dmapjoint0, dmapjoint1;

  for (std::size_t index = 0; index < facets.size(); ++index)
  {
    const std::array<std::int32_t, 2> cells
        = {std::get<0>(facets[index]), std::get<2>(facets[index])};
    const std::array<int, 2> local_facet
        = {std::get<1>(facets[index]), std::get<3>(facets[index])};

    // Get cell geometry
    auto x_dofs0 = x_dofmap.links(cells[0]);
    for (std::size_t i = 0; i < x_dofs0.size(); ++i)
    {
      common::impl::copy_N<3>(
          std::next(x_g.begin(), 3 * x_dofs0[i]),
          xt::view(coordinate_dofs, 0, i, xt::all()).begin());
    }
    auto x_dofs1 = x_dofmap.links(cells[1]);
    for (std::size_t i = 0; i < x_dofs1.size(); ++i)
    {
      common::impl::copy_N<3>(
          std::next(x_g.begin(), 3 * x_dofs1[i]),
          xt::view(coordinate_dofs, 1, i, xt::all()).begin());
    }

    // Get dof maps for cells and pack
    xtl::span<const std::int32_t> dmap0_cell0 = dofmap0.cell_dofs(cells[0]);
    xtl::span<const std::int32_t> dmap0_cell1 = dofmap0.cell_dofs(cells[1]);
    dmapjoint0.resize(dmap0_cell0.size() + dmap0_cell1.size());
    std::copy(dmap0_cell0.begin(), dmap0_cell0.end(), dmapjoint0.begin());
    std::copy(dmap0_cell1.begin(), dmap0_cell1.end(),
              std::next(dmapjoint0.begin(), dmap0_cell0.size()));

    xtl::span<const std::int32_t> dmap1_cell0 = dofmap1.cell_dofs(cells[0]);
    xtl::span<const std::int32_t> dmap1_cell1 = dofmap1.cell_dofs(cells[1]);
    dmapjoint1.resize(dmap1_cell0.size() + dmap1_cell1.size());
    std::copy(dmap1_cell0.begin(), dmap1_cell0.end(), dmapjoint1.begin());
    std::copy(dmap1_cell1.begin(), dmap1_cell1.end(),
              std::next(dmapjoint1.begin(), dmap1_cell0.size()));

    const int num_rows = bs0 * dmapjoint0.size();
    const int num_cols = bs1 * dmapjoint1.size();

    // Tabulate tensor
    Ae.resize(num_rows * num_cols);
    std::fill(Ae.begin(), Ae.end(), 0);

    const std::array perm{
        get_perm(cells[0] * num_cell_facets + local_facet[0]),
        get_perm(cells[1] * num_cell_facets + local_facet[1])};
    kernel(Ae.data(), coeffs.data() + index * 2 * cstride, constants.data(),
           coordinate_dofs.data(), local_facet.data(), perm.data());

    const xtl::span<T> _Ae(Ae);

    const xtl::span<T> sub_Ae0
        = _Ae.subspan(bs0 * dmap0_cell0.size() * num_cols,
                      bs0 * dmap0_cell1.size() * num_cols);
    const xtl::span<T> sub_Ae1
        = _Ae.subspan(bs1 * dmap1_cell0.size(),
                      num_rows * num_cols - bs1 * dmap1_cell0.size());

    // Need to apply DOF transformations for parts of the matrix due to cell 0
    // and cell 1. For example, if the space has 3 DOFs, then Ae will be 6 by 6
    // (3 rows/columns for each cell). Subspans are used to offset to the right
    // blocks of the matrix

    dof_transform(_Ae, cell_info, cells[0], num_cols);
    dof_transform(sub_Ae0, cell_info, cells[1], num_cols);
    dof_transform_to_transpose(_Ae, cell_info, cells[0], num_rows);
    dof_transform_to_transpose(sub_Ae1, cell_info, cells[1], num_rows);

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

    mat_set(dmapjoint0, dmapjoint1, Ae);
  }
}

template <typename T, typename U>
void assemble_matrix(
    U mat_set, const Form<T>& a, const xtl::span<const T>& constants,
    const std::map<std::pair<IntegralType, int>,
                   std::pair<xtl::span<const T>, int>>& coefficients,
    const xtl::span<const std::int8_t>& bc0,
    const xtl::span<const std::int8_t>& bc1)
{
  std::shared_ptr<const mesh::Mesh> mesh = a.mesh();
  assert(mesh);

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

  std::shared_ptr<const fem::FiniteElement> element0
      = a.function_spaces().at(0)->element();
  std::shared_ptr<const fem::FiniteElement> element1
      = a.function_spaces().at(1)->element();
  const std::function<void(const xtl::span<T>&,
                           const xtl::span<const std::uint32_t>&, std::int32_t,
                           int)>& dof_transform
      = element0->get_dof_transformation_function<T>();
  const std::function<void(const xtl::span<T>&,
                           const xtl::span<const std::uint32_t>&, std::int32_t,
                           int)>& dof_transform_to_transpose
      = element1->get_dof_transformation_to_transpose_function<T>();

  const bool needs_transformation_data
      = element0->needs_dof_transformations()
        or element1->needs_dof_transformations()
        or a.needs_facet_permutations();
  xtl::span<const std::uint32_t> cell_info;
  if (needs_transformation_data)
  {
    mesh->topology_mutable().create_entity_permutations();
    cell_info = xtl::span(mesh->topology().get_cell_permutation_info());
  }

  for (int i : a.integral_ids(IntegralType::cell))
  {
    const auto& fn = a.kernel(IntegralType::cell, i);
    const auto& [coeffs, cstride] = coefficients.at({IntegralType::cell, i});
    const std::vector<std::int32_t>& cells = a.cell_domains(i);
    impl::assemble_cells(mat_set, mesh->geometry(), cells, dof_transform, dofs0,
                         bs0, dof_transform_to_transpose, dofs1, bs1, bc0, bc1,
                         fn, coeffs, cstride, constants, cell_info);
  }

  for (int i : a.integral_ids(IntegralType::exterior_facet))
  {
    const auto& fn = a.kernel(IntegralType::exterior_facet, i);
    const auto& [coeffs, cstride]
        = coefficients.at({IntegralType::exterior_facet, i});
    const std::vector<std::pair<std::int32_t, int>>& facets
        = a.exterior_facet_domains(i);
    impl::assemble_exterior_facets(mat_set, *mesh, facets, dof_transform, dofs0,
                                   bs0, dof_transform_to_transpose, dofs1, bs1,
                                   bc0, bc1, fn, coeffs, cstride, constants,
                                   cell_info);
  }

  if (a.num_integrals(IntegralType::interior_facet) > 0)
  {
    std::function<std::uint8_t(std::size_t)> get_perm;
    if (a.needs_facet_permutations())
    {
      mesh->topology_mutable().create_entity_permutations();
      const std::vector<std::uint8_t>& perms
          = mesh->topology().get_facet_permutations();
      get_perm = [&perms](std::size_t i) { return perms[i]; };
    }
    else
      get_perm = [](std::size_t) { return 0; };

    const std::vector<int> c_offsets = a.coefficient_offsets();
    for (int i : a.integral_ids(IntegralType::interior_facet))
    {
      const auto& fn = a.kernel(IntegralType::interior_facet, i);
      const auto& [coeffs, cstride]
          = coefficients.at({IntegralType::interior_facet, i});
      const std::vector<std::tuple<std::int32_t, int, std::int32_t, int>>&
          facets
          = a.interior_facet_domains(i);
      impl::assemble_interior_facets(
          mat_set, *mesh, facets, dof_transform, *dofmap0, bs0,
          dof_transform_to_transpose, *dofmap1, bs1, bc0, bc1, fn, coeffs,
          cstride, c_offsets, constants, cell_info, get_perm);
    }
  }
}

} // namespace dolfinx::fem::impl
