// Copyright (C) 2018-2019 Garth N. Wells
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include "DofMap.h"
#include "Form.h"
#include "FunctionSpace.h"
#include "traits.h"
#include "utils.h"
#include <algorithm>
#include <dolfinx/la/utils.h>
#include <dolfinx/mesh/Geometry.h>
#include <dolfinx/mesh/Mesh.h>
#include <dolfinx/mesh/Topology.h>
#include <functional>
#include <iterator>
#include <span>
#include <tuple>
#include <vector>

namespace dolfinx::fem::impl
{

using mdspan2_t = MDSPAN_IMPL_STANDARD_NAMESPACE::mdspan<
    const std::int32_t,
    MDSPAN_IMPL_STANDARD_NAMESPACE::dextents<std::size_t, 2>>;

/// @brief Execute kernel over cells and accumulate result in matrix.
/// @tparam T Matrix/form scalar type.
/// @param mat_set Function that accumulates computed entries into a
/// matrix.
/// @param x_dofmap Dofmap for the mesh geometry.
/// @param x Mesh geometry (coordinates).
/// @param cells Cell indices (in the integration domain mesh) to execute
/// the kernel over. These are the indices into the geometry dofmap.
/// @param dofmap0 Test function (row) degree-of-freedom data holding
/// the (0) dofmap, (1) dofmap block size and (2) dofmap cell indices.
/// @param P0 Function that applies transformation P_0 A in-place to
/// transform trial degrees-of-freedom.
/// @param dofmap1 Trial function (column) degree-of-freedom data
/// holding the (0) dofmap, (1) dofmap block size and (2) dofmap cell
/// indices.
/// @param P1T Function that applies transformation A P_1^T in-place to
/// transform trial degrees-of-freedom.
/// @param bc0 Marker for rows with Dirichlet boundary conditions applied
/// @param bc1 Marker for columns with Dirichlet boundary conditions applied
/// @param kernel Kernel function to execute over each cell.
/// @param coeffs The coefficient data array of shape (cells.size(), cstride),
/// flattened into row-major format.
/// @param cstride The coefficient stride
/// @param constants The constant data
/// @param cell_info0 The cell permutation information for the test function
/// mesh
/// @param cell_info1 The cell permutation information for the trial function
/// mesh
template <dolfinx::scalar T>
void assemble_cells(
    la::MatSet<T> auto mat_set, mdspan2_t x_dofmap,
    std::span<const scalar_value_type_t<T>> x,
    std::span<const std::int32_t> cells,
    std::tuple<mdspan2_t, int, std::span<const std::int32_t>> dofmap0,
    fem::DofTransformKernel<T> auto P0,
    std::tuple<mdspan2_t, int, std::span<const std::int32_t>> dofmap1,
    fem::DofTransformKernel<T> auto P1T, std::span<const std::int8_t> bc0,
    std::span<const std::int8_t> bc1, FEkernel<T> auto kernel,
    std::span<const T> coeffs, int cstride, std::span<const T> constants,
    std::span<const std::uint32_t> cell_info0,
    std::span<const std::uint32_t> cell_info1)
{
  if (cells.empty())
    return;

  const auto [dmap0, bs0, cells0] = dofmap0;
  const auto [dmap1, bs1, cells1] = dofmap1;

  // Iterate over active cells
  const int num_dofs0 = dmap0.extent(1);
  const int num_dofs1 = dmap1.extent(1);
  const int ndim0 = bs0 * num_dofs0;
  const int ndim1 = bs1 * num_dofs1;
  std::vector<T> Ae(ndim0 * ndim1);
  std::span<T> _Ae(Ae);
  std::vector<scalar_value_type_t<T>> coordinate_dofs(3 * x_dofmap.extent(1));

  // Iterate over active cells
  assert(cells0.size() == cells.size());
  assert(cells1.size() == cells.size());
  for (std::size_t index = 0; index < cells.size(); ++index)
  {
    // Cell index in integration domain mesh (c), test function mesh
    // (c0) and trial function mesh (c1)
    std::int32_t c = cells[index];
    std::int32_t c0 = cells0[index];
    std::int32_t c1 = cells1[index];

    // Get cell coordinates/geometry
    auto x_dofs
        = MDSPAN_IMPL_STANDARD_NAMESPACE::MDSPAN_IMPL_PROPOSED_NAMESPACE::
            submdspan(x_dofmap, c, MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent);
    for (std::size_t i = 0; i < x_dofs.size(); ++i)
    {
      std::copy_n(std::next(x.begin(), 3 * x_dofs[i]), 3,
                  std::next(coordinate_dofs.begin(), 3 * i));
    }

    // Tabulate tensor
    std::fill(Ae.begin(), Ae.end(), 0);
    kernel(Ae.data(), coeffs.data() + index * cstride, constants.data(),
           coordinate_dofs.data(), nullptr, nullptr);

    // Compute A = P_0 \tilde{A} P_1^T (dof transformation)
    P0(_Ae, cell_info1, c1, ndim1);  // B = P0 \tilde{A}
    P1T(_Ae, cell_info0, c0, ndim0); // A =  B P1_T

    // Zero rows/columns for essential bcs
    auto dofs0 = std::span(dmap0.data_handle() + c0 * num_dofs0, num_dofs0);
    auto dofs1 = std::span(dmap1.data_handle() + c1 * num_dofs1, num_dofs1);

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
            std::fill_n(std::next(Ae.begin(), ndim1 * row), ndim1, 0);
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
              Ae[row * ndim1 + col] = 0;
          }
        }
      }
    }

    mat_set(dofs0, dofs1, Ae);
  }
}

/// Execute kernel over exterior facets and  accumulate result in Mat
template <dolfinx::scalar T>
void assemble_exterior_facets(
    la::MatSet<T> auto mat_set, mdspan2_t x_dofmap,
    std::span<const scalar_value_type_t<T>> x,
    std::span<const std::int32_t> facets,
    std::tuple<mdspan2_t, int, std::span<const std::int32_t>> dofmap0,
    fem::DofTransformKernel<T> auto P0,
    std::tuple<mdspan2_t, int, std::span<const std::int32_t>> dofmap1,
    fem::DofTransformKernel<T> auto P1T, std::span<const std::int8_t> bc0,
    std::span<const std::int8_t> bc1, FEkernel<T> auto kernel,
    std::span<const T> coeffs, int cstride, std::span<const T> constants,
    std::span<const std::uint32_t> cell_info0,
    std::span<const std::uint32_t> cell_info1)
{
  if (facets.empty())
    return;

  const auto [dmap0, bs0, facets0] = dofmap0;
  const auto [dmap1, bs1, facets1] = dofmap1;

  // Data structures used in assembly
  std::vector<scalar_value_type_t<T>> coordinate_dofs(3 * x_dofmap.extent(1));
  const int num_dofs0 = dmap0.extent(1);
  const int num_dofs1 = dmap1.extent(1);
  const int ndim0 = bs0 * num_dofs0;
  const int ndim1 = bs1 * num_dofs1;
  std::vector<T> Ae(ndim0 * ndim1);
  std::span<T> _Ae(Ae);
  assert(facets.size() % 2 == 0);
  assert(facets0.size() == facets.size());
  assert(facets1.size() == facets.size());
  for (std::size_t index = 0; index < facets.size(); index += 2)
  {
    // Cell in the integration domain
    std::int32_t cell = facets[index];
    // Cell in the test function mesh
    std::int32_t cell0 = facets0[index];
    // Cell in the trial function mesh
    std::int32_t cell1 = facets1[index];
    std::int32_t local_facet = facets[index + 1];

    // Get cell coordinates/geometry
    auto x_dofs = MDSPAN_IMPL_STANDARD_NAMESPACE::
        MDSPAN_IMPL_PROPOSED_NAMESPACE::submdspan(
            x_dofmap, cell, MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent);
    for (std::size_t i = 0; i < x_dofs.size(); ++i)
    {
      std::copy_n(std::next(x.begin(), 3 * x_dofs[i]), 3,
                  std::next(coordinate_dofs.begin(), 3 * i));
    }

    // Tabulate tensor
    std::fill(Ae.begin(), Ae.end(), 0);
    kernel(Ae.data(), coeffs.data() + index / 2 * cstride, constants.data(),
           coordinate_dofs.data(), &local_facet, nullptr);

    P0(_Ae, cell_info1, cell1, ndim1);
    P1T(_Ae, cell_info0, cell0, ndim0);

    // Zero rows/columns for essential bcs
    auto dofs0 = std::span(dmap0.data_handle() + cell0 * num_dofs0, num_dofs0);
    auto dofs1 = std::span(dmap1.data_handle() + cell1 * num_dofs1, num_dofs1);
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
            std::fill_n(std::next(Ae.begin(), ndim1 * row), ndim1, 0);
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
              Ae[row * ndim1 + col] = 0;
          }
        }
      }
    }

    mat_set(dofs0, dofs1, Ae);
  }
}

/// Execute kernel over interior facets and  accumulate result in Mat
template <dolfinx::scalar T>
void assemble_interior_facets(
    la::MatSet<T> auto mat_set, mdspan2_t x_dofmap,
    std::span<const scalar_value_type_t<T>> x, int num_cell_facets,
    std::span<const std::int32_t> facets, fem::DofTransformKernel<T> auto P0,
    const DofMap& dofmap0, int bs0, fem::DofTransformKernel<T> auto P1T,
    const DofMap& dofmap1, int bs1, std::span<const std::int8_t> bc0,
    std::span<const std::int8_t> bc1, FEkernel<T> auto kernel,
    std::span<const T> coeffs, int cstride, std::span<const int> offsets,
    std::span<const T> constants, std::span<const std::uint32_t> cell_info,
    const std::function<std::uint8_t(std::size_t)>& get_perm)
{
  if (facets.empty())
    return;

  // Data structures used in assembly
  using X = scalar_value_type_t<T>;
  std::vector<X> coordinate_dofs(2 * x_dofmap.extent(1) * 3);
  std::span<X> cdofs0(coordinate_dofs.data(), x_dofmap.extent(1) * 3);
  std::span<X> cdofs1(coordinate_dofs.data() + x_dofmap.extent(1) * 3,
                      x_dofmap.extent(1) * 3);

  std::vector<T> Ae, be;
  std::vector<T> coeff_array(2 * offsets.back());
  assert(offsets.back() == cstride);

  // Temporaries for joint dofmaps
  std::vector<std::int32_t> dmapjoint0, dmapjoint1;
  assert(facets.size() % 4 == 0);
  for (std::size_t index = 0; index < facets.size(); index += 4)
  {
    std::array<std::int32_t, 2> cells = {facets[index], facets[index + 2]};
    std::array<std::int32_t, 2> local_facet
        = {facets[index + 1], facets[index + 3]};

    // Get cell geometry
    auto x_dofs0 = MDSPAN_IMPL_STANDARD_NAMESPACE::
        MDSPAN_IMPL_PROPOSED_NAMESPACE::submdspan(
            x_dofmap, cells[0], MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent);
    for (std::size_t i = 0; i < x_dofs0.size(); ++i)
    {
      std::copy_n(std::next(x.begin(), 3 * x_dofs0[i]), 3,
                  std::next(cdofs0.begin(), 3 * i));
    }
    auto x_dofs1 = MDSPAN_IMPL_STANDARD_NAMESPACE::
        MDSPAN_IMPL_PROPOSED_NAMESPACE::submdspan(
            x_dofmap, cells[1], MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent);
    for (std::size_t i = 0; i < x_dofs1.size(); ++i)
    {
      std::copy_n(std::next(x.begin(), 3 * x_dofs1[i]), 3,
                  std::next(cdofs1.begin(), 3 * i));
    }

    // Get dof maps for cells and pack
    std::span<const std::int32_t> dmap0_cell0 = dofmap0.cell_dofs(cells[0]);
    std::span<const std::int32_t> dmap0_cell1 = dofmap0.cell_dofs(cells[1]);
    dmapjoint0.resize(dmap0_cell0.size() + dmap0_cell1.size());
    std::copy(dmap0_cell0.begin(), dmap0_cell0.end(), dmapjoint0.begin());
    std::copy(dmap0_cell1.begin(), dmap0_cell1.end(),
              std::next(dmapjoint0.begin(), dmap0_cell0.size()));

    std::span<const std::int32_t> dmap1_cell0 = dofmap1.cell_dofs(cells[0]);
    std::span<const std::int32_t> dmap1_cell1 = dofmap1.cell_dofs(cells[1]);
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
    kernel(Ae.data(), coeffs.data() + index / 2 * cstride, constants.data(),
           coordinate_dofs.data(), local_facet.data(), perm.data());

    // Local element layout is a 2x2 block matrix with structure
    //
    //   cell0cell0  |  cell0cell1
    //   cell1cell0  |  cell1cell1
    //
    // where each block is element tensor of size (dmap0, dmap1).

    std::span<T> _Ae(Ae);
    std::span<T> sub_Ae0 = _Ae.subspan(bs0 * dmap0_cell0.size() * num_cols,
                                       bs0 * dmap0_cell1.size() * num_cols);

    P0(_Ae, cell_info, cells[0], num_cols);
    P0(sub_Ae0, cell_info, cells[1], num_cols);
    P1T(_Ae, cell_info, cells[0], num_rows);

    for (int row = 0; row < num_rows; ++row)
    {
      // DOFs for dmap1 and cell1 are not stored contiguously in
      // the block matrix, so each row needs a separate span access
      std::span<T> sub_Ae1 = _Ae.subspan(
          row * num_cols + bs1 * dmap1_cell0.size(), bs1 * dmap1_cell1.size());
      P1T(sub_Ae1, cell_info, cells[1], 1);
    }

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
                        num_cols, 0);
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
              Ae[m * num_cols + bs1 * j + k] = 0;
          }
        }
      }
    }

    mat_set(dmapjoint0, dmapjoint1, Ae);
  }
}

/// The matrix A must already be initialised. The matrix may be a proxy,
/// i.e. a view into a larger matrix, and assembly is performed using
/// local indices. Rows (bc0) and columns (bc1) with Dirichlet
/// conditions are zeroed. Markers (bc0 and bc1) can be empty if no bcs
/// are applied. Matrix is not finalised.
template <dolfinx::scalar T, std::floating_point U>
void assemble_matrix(
    la::MatSet<T> auto mat_set, const Form<T, U>& a, mdspan2_t x_dofmap,
    std::span<const scalar_value_type_t<T>> x, std::span<const T> constants,
    const std::map<std::pair<IntegralType, int>,
                   std::pair<std::span<const T>, int>>& coefficients,
    std::span<const std::int8_t> bc0, std::span<const std::int8_t> bc1)
{
  // Integration domain mesh
  std::shared_ptr<const mesh::Mesh<U>> mesh = a.mesh();
  assert(mesh);
  // Test function mesh
  auto mesh0 = a.function_spaces().at(0)->mesh();
  assert(mesh0);
  // Trial function mesh
  auto mesh1 = a.function_spaces().at(1)->mesh();
  assert(mesh1);

  // Get dofmap data
  std::shared_ptr<const fem::DofMap> dofmap0
      = a.function_spaces().at(0)->dofmap();
  std::shared_ptr<const fem::DofMap> dofmap1
      = a.function_spaces().at(1)->dofmap();
  assert(dofmap0);
  assert(dofmap1);
  auto dofs0 = dofmap0->map();
  const int bs0 = dofmap0->bs();
  auto dofs1 = dofmap1->map();
  const int bs1 = dofmap1->bs();

  auto element0 = a.function_spaces().at(0)->element();
  assert(element0);
  auto element1 = a.function_spaces().at(1)->element();
  assert(element1);
  fem::DofTransformKernel<T> auto P0
      = element0->template get_pre_dof_transformation_function<T>();
  fem::DofTransformKernel<T> auto P1T
      = element1->template get_post_dof_transformation_function<T>(
          FiniteElement<U>::doftransform::transpose);

  std::span<const std::uint32_t> cell_info0;
  std::span<const std::uint32_t> cell_info1;
  if (element0->needs_dof_transformations()
      or element1->needs_dof_transformations() or a.needs_facet_permutations())
  {
    mesh0->topology_mutable()->create_entity_permutations();
    mesh1->topology_mutable()->create_entity_permutations();
    cell_info0 = std::span(mesh0->topology()->get_cell_permutation_info());
    cell_info1 = std::span(mesh1->topology()->get_cell_permutation_info());
  }

  for (int i : a.integral_ids(IntegralType::cell))
  {
    auto fn = a.kernel(IntegralType::cell, i);
    assert(fn);
    auto& [coeffs, cstride] = coefficients.at({IntegralType::cell, i});
    impl::assemble_cells(
        mat_set, x_dofmap, x, a.domain(IntegralType::cell, i),
        {dofs0, bs0, a.domain(IntegralType::cell, i, *mesh0)}, P0,
        {dofs1, bs1, a.domain(IntegralType::cell, i, *mesh1)}, P1T, bc0, bc1,
        fn, coeffs, cstride, constants, cell_info0, cell_info1);
  }

  for (int i : a.integral_ids(IntegralType::exterior_facet))
  {
    auto fn = a.kernel(IntegralType::exterior_facet, i);
    assert(fn);
    auto& [coeffs, cstride]
        = coefficients.at({IntegralType::exterior_facet, i});
    impl::assemble_exterior_facets(
        mat_set, x_dofmap, x, a.domain(IntegralType::exterior_facet, i),
        {dofs0, bs0, a.domain(IntegralType::exterior_facet, i, *mesh0)}, P0,
        {dofs1, bs1, a.domain(IntegralType::exterior_facet, i, *mesh1)}, P1T,
        bc0, bc1, fn, coeffs, cstride, constants, cell_info0, cell_info1);
  }

  if (a.num_integrals(IntegralType::interior_facet) > 0)
  {
    std::function<std::uint8_t(std::size_t)> get_perm;
    if (a.needs_facet_permutations())
    {
      mesh->topology_mutable()->create_entity_permutations();
      const std::vector<std::uint8_t>& perms
          = mesh->topology()->get_facet_permutations();
      get_perm = [&perms](std::size_t i) { return perms[i]; };
    }
    else
      get_perm = [](std::size_t) { return 0; };

    mesh::CellType cell_type = mesh->topology()->cell_type();
    int num_cell_facets
        = mesh::cell_num_entities(cell_type, mesh->topology()->dim() - 1);
    const std::vector<int> c_offsets = a.coefficient_offsets();
    for (int i : a.integral_ids(IntegralType::interior_facet))
    {
      auto fn = a.kernel(IntegralType::interior_facet, i);
      assert(fn);
      auto& [coeffs, cstride]
          = coefficients.at({IntegralType::interior_facet, i});
      impl::assemble_interior_facets(mat_set, x_dofmap, x, num_cell_facets,
                                     a.domain(IntegralType::interior_facet, i),
                                     P0, *dofmap0, bs0, P1T, *dofmap1, bs1, bc0,
                                     bc1, fn, coeffs, cstride, c_offsets,
                                     constants, cell_info0, get_perm);
    }
  }
}

} // namespace dolfinx::fem::impl
