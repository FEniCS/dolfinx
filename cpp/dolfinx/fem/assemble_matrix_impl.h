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
namespace md = MDSPAN_IMPL_STANDARD_NAMESPACE;

/// @brief Typedef
using mdspan2_t = md::mdspan<const std::int32_t, md::dextents<std::size_t, 2>>;

/// @brief Execute kernel over cells and accumulate result in a matrix.
///
/// @tparam T Matrix/form scalar type.
/// @param mat_set Function that accumulates computed entries into a
/// matrix.
/// @param[in] x_dofmap Degree-of-freedom map for the mesh geometry.
/// @param[in] x Mesh geometry (coordinates).
/// @param[in] cells Cell indices to execute the kernel over. These are
/// the indices into the geometry dofmap `x_dofmap`.
/// @param[in] dofmap0 Test function (row) degree-of-freedom data
/// holding the (0) dofmap, (1) dofmap block size and (2) dofmap cell
/// indices.
/// @param[in] P0 Function that applies transformation `P_0 A` in-place
/// to the computed tensor `A` to transform its test degrees-of-freedom.
/// @param[in] dofmap1 Trial function (column) degree-of-freedom data
/// holding the (0) dofmap, (1) dofmap block size and (2) dofmap cell
/// indices.
/// @param[in] P1T Function that applies transformation `A P_1^T`
/// in-place to to the computed tensor `A` to transform trial
/// degrees-of-freedom.
/// @param bc0 Marker for rows with Dirichlet boundary conditions
/// applied.
/// @param bc1 Marker for columns with Dirichlet boundary conditions
/// applied.
/// @param kernel Kernel function to execute over each cell.
/// @param[in] coeffs Coefficient data in the kernel. It has shape
/// `(cells.size(), num_cell_coeffs)`. `coeffs(i, j)` is the `j`th
/// coefficient for cell `i`.
/// @param constants Constant data.
/// @param cell_info0 Cell permutation information for the test
/// function mesh.
/// @param cell_info1 Cell permutation information for the trial
/// function mesh.
template <dolfinx::scalar T>
void assemble_cells(
    la::MatSet<T> auto mat_set, mdspan2_t x_dofmap,
    md::mdspan<const scalar_value_t<T>,
               md::extents<std::size_t, md::dynamic_extent, 3>>
        x,
    std::span<const std::int32_t> cells,
    std::tuple<mdspan2_t, int, std::span<const std::int32_t>> dofmap0,
    fem::DofTransformKernel<T> auto P0,
    std::tuple<mdspan2_t, int, std::span<const std::int32_t>> dofmap1,
    fem::DofTransformKernel<T> auto P1T, std::span<const std::int8_t> bc0,
    std::span<const std::int8_t> bc1, FEkernel<T> auto kernel,
    md::mdspan<const T, md::dextents<std::size_t, 2>> coeffs,
    std::span<const T> constants, std::span<const std::uint32_t> cell_info0,
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
  std::vector<scalar_value_t<T>> cdofs(3 * x_dofmap.extent(1));

  // Iterate over active cells
  assert(cells0.size() == cells.size());
  assert(cells1.size() == cells.size());
  for (std::size_t c = 0; c < cells.size(); ++c)
  {
    // Cell index in integration domain mesh (c), test function mesh
    // (c0) and trial function mesh (c1)
    std::int32_t cell = cells[c];
    std::int32_t cell0 = cells0[c];
    std::int32_t cell1 = cells1[c];

    // Get cell coordinates/geometry
    auto x_dofs = md::submdspan(x_dofmap, cell, md::full_extent);
    for (std::size_t i = 0; i < x_dofs.size(); ++i)
      std::copy_n(&x(x_dofs[i], 0), 3, std::next(cdofs.begin(), 3 * i));

    // Tabulate tensor
    std::ranges::fill(Ae, 0);
    kernel(Ae.data(), &coeffs(c, 0), constants.data(), cdofs.data(), nullptr,
           nullptr);

    // Compute A = P_0 \tilde{A} P_1^T (dof transformation)
    P0(_Ae, cell_info0, cell0, ndim1);  // B = P0 \tilde{A}
    P1T(_Ae, cell_info1, cell1, ndim0); // A =  B P1_T

    // Zero rows/columns for essential bcs
    std::span dofs0(dmap0.data_handle() + cell0 * num_dofs0, num_dofs0);
    std::span dofs1(dmap1.data_handle() + cell1 * num_dofs1, num_dofs1);

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

/// @brief Execute kernel over exterior facets and accumulate result in
/// a matrix.
///
/// @tparam T Matrix/form scalar type.
/// @param[in] mat_set Function that accumulates computed entries into a
/// matrix.
/// @param[in] x_dofmap Dofmap for the mesh geometry.
/// @param[in] x Mesh geometry (coordinates).
/// @param[in] facets Facet indices (in the integration domain mesh) to
/// execute the kernel over.
/// @param[in] dofmap0 Test function (row) degree-of-freedom data
/// holding the (0) dofmap, (1) dofmap block size and (2) dofmap cell
/// indices.
/// @param[in] P0 Function that applies transformation P0.A in-place to
/// transform test degrees-of-freedom.
/// @param[in] dofmap1 Trial function (column) degree-of-freedom data
/// holding the (0) dofmap, (1) dofmap block size and (2) dofmap cell
/// indices.
/// @param[in] P1T Function that applies transformation A.P1^T in-place
/// to transform trial degrees-of-freedom.
/// @param[in] bc0 Marker for rows with Dirichlet boundary conditions
/// applied.
/// @param[in] bc1 Marker for columns with Dirichlet boundary conditions
/// applied.
/// @param[in] kernel Kernel function to execute over each cell.
/// @param[in] coeffs Coefficient data array of shape `(cells.size(),
/// cstride)`.
/// @param[in] constants Constant data.
/// @param[in] cell_info0 Cell permutation information for the test
/// function mesh.
/// @param[in] cell_info1 Cell permutation information for the trial
/// function mesh.
/// @param[in] perms Facet permutation integer. Empty if facet
/// permutations are not required.
template <dolfinx::scalar T>
void assemble_exterior_facets(
    la::MatSet<T> auto mat_set, mdspan2_t x_dofmap,
    md::mdspan<const scalar_value_t<T>,
               md::extents<std::size_t, md::dynamic_extent, 3>>
        x,
    md::mdspan<const std::int32_t,
               std::extents<std::size_t, md::dynamic_extent, 2>>
        facets,
    std::tuple<mdspan2_t, int,
               md::mdspan<const std::int32_t,
                          std::extents<std::size_t, md::dynamic_extent, 2>>>
        dofmap0,
    fem::DofTransformKernel<T> auto P0,
    std::tuple<mdspan2_t, int,
               md::mdspan<const std::int32_t,
                          std::extents<std::size_t, md::dynamic_extent, 2>>>
        dofmap1,
    fem::DofTransformKernel<T> auto P1T, std::span<const std::int8_t> bc0,
    std::span<const std::int8_t> bc1, FEkernel<T> auto kernel,
    md::mdspan<const T, md::dextents<std::size_t, 2>> coeffs,
    std::span<const T> constants, std::span<const std::uint32_t> cell_info0,
    std::span<const std::uint32_t> cell_info1,
    md::mdspan<const std::uint8_t, md::dextents<std::size_t, 2>> perms)
{
  if (facets.empty())
    return;

  const auto [dmap0, bs0, facets0] = dofmap0;
  const auto [dmap1, bs1, facets1] = dofmap1;

  // Data structures used in assembly
  std::vector<scalar_value_t<T>> cdofs(3 * x_dofmap.extent(1));
  const int num_dofs0 = dmap0.extent(1);
  const int num_dofs1 = dmap1.extent(1);
  const int ndim0 = bs0 * num_dofs0;
  const int ndim1 = bs1 * num_dofs1;
  std::vector<T> Ae(ndim0 * ndim1);
  std::span<T> _Ae(Ae);
  assert(facets0.size() == facets.size());
  assert(facets1.size() == facets.size());
  for (std::size_t f = 0; f < facets.extent(0); ++f)
  {
    // Cell in the integration domain, local facet index relative to the
    // integration domain cell, and cells in the test and trial function
    // meshes
    std::int32_t cell = facets(f, 0);
    std::int32_t local_facet = facets(f, 1);
    std::int32_t cell0 = facets0(f, 0);
    std::int32_t cell1 = facets1(f, 0);

    // Get cell coordinates/geometry
    auto x_dofs = md::submdspan(x_dofmap, cell, md::full_extent);
    for (std::size_t i = 0; i < x_dofs.size(); ++i)
      std::copy_n(&x(x_dofs[i], 0), 3, std::next(cdofs.begin(), 3 * i));

    // Permutations
    std::uint8_t perm = perms.empty() ? 0 : perms(cell, local_facet);

    // Tabulate tensor
    std::ranges::fill(Ae, 0);
    kernel(Ae.data(), &coeffs(f, 0), constants.data(), cdofs.data(),
           &local_facet, &perm);

    P0(_Ae, cell_info0, cell0, ndim1);
    P1T(_Ae, cell_info1, cell1, ndim0);

    // Zero rows/columns for essential bcs
    std::span dofs0(dmap0.data_handle() + cell0 * num_dofs0, num_dofs0);
    std::span dofs1(dmap1.data_handle() + cell1 * num_dofs1, num_dofs1);
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

/// @brief Execute kernel over interior facets and accumulate result in
/// a matrix.
///
/// @tparam T Matrix/form scalar type.
/// @param mat_set Function that accumulates computed entries into a
/// matrix.
/// @param[in] x_dofmap Dofmap for the mesh geometry.
/// @param[in] x Mesh geometry (coordinates).
/// @param[in] facets Facet indices (in the integration domain mesh) to
/// execute the kernel over.
/// @param[in] dofmap0 Test function (row) degree-of-freedom data
/// holding the (0) dofmap, (1) dofmap block size and (2) dofmap cell
/// indices.
/// @param[in] P0 Function that applies transformation P0.A in-place to
/// transform test degrees-of-freedom.
/// @param[in] dofmap1 Trial function (column) degree-of-freedom data
/// holding the (0) dofmap, (1) dofmap block size and (2) dofmap cell
/// indices.
/// @param[in] P1T Function that applies transformation A.P1^T in-place
/// to transform trial degrees-of-freedom.
/// @param[in] bc0 Marker for rows with Dirichlet boundary conditions
/// applied.
/// @param[in] bc1 Marker for columns with Dirichlet boundary conditions
/// applied.
/// @param[in] kernel Kernel function to execute over each cell.
/// @param[in] coeffs  The coefficient data array of shape (cells.size(),
/// cstride).
/// @param[in] constants Constant data.
/// @param[in] cell_info0 Cell permutation information for the test
/// function mesh.
/// @param[in] cell_info1 Cell permutation information for the trial
/// function mesh.
/// @param[in] perms Facet permutation integer. Empty if facet
/// permutations are not required.
template <dolfinx::scalar T>
void assemble_interior_facets(
    la::MatSet<T> auto mat_set, mdspan2_t x_dofmap,
    md::mdspan<const scalar_value_t<T>,
               md::extents<std::size_t, md::dynamic_extent, 3>>
        x,
    md::mdspan<const std::int32_t,
               std::extents<std::size_t, md::dynamic_extent, 2, 2>>
        facets,
    std::tuple<const DofMap&, int,
               md::mdspan<const std::int32_t,
                          std::extents<std::size_t, md::dynamic_extent, 2, 2>>>
        dofmap0,
    fem::DofTransformKernel<T> auto P0,
    std::tuple<const DofMap&, int,
               md::mdspan<const std::int32_t,
                          std::extents<std::size_t, md::dynamic_extent, 2, 2>>>
        dofmap1,
    fem::DofTransformKernel<T> auto P1T, std::span<const std::int8_t> bc0,
    std::span<const std::int8_t> bc1, FEkernel<T> auto kernel,
    md::mdspan<const T, md::extents<std::size_t, md::dynamic_extent, 2,
                                    md::dynamic_extent>>
        coeffs,
    std::span<const T> constants, std::span<const std::uint32_t> cell_info0,
    std::span<const std::uint32_t> cell_info1,
    md::mdspan<const std::uint8_t, md::dextents<std::size_t, 2>> perms)
{
  if (facets.empty())
    return;

  const auto [dmap0, bs0, facets0] = dofmap0;
  const auto [dmap1, bs1, facets1] = dofmap1;

  // Data structures used in assembly
  using X = scalar_value_t<T>;
  std::vector<X> cdofs(2 * x_dofmap.extent(1) * 3);
  std::span<X> cdofs0(cdofs.data(), x_dofmap.extent(1) * 3);
  std::span<X> cdofs1(cdofs.data() + x_dofmap.extent(1) * 3,
                      x_dofmap.extent(1) * 3);

  std::vector<T> Ae, be;

  // Temporaries for joint dofmaps
  std::vector<std::int32_t> dmapjoint0, dmapjoint1;
  assert(facets0.size() == facets.size());
  assert(facets1.size() == facets.size());
  for (std::size_t f = 0; f < facets.extent(0); ++f)
  {
    // Cells in integration domain,  test function domain and trial
    // function domain
    std::array cells{facets(f, 0, 0), facets(f, 1, 0)};
    std::array cells0{facets0(f, 0, 0), facets0(f, 1, 0)};
    std::array cells1{facets1(f, 0, 0), facets1(f, 1, 0)};

    // Local facets indices
    std::array local_facet{facets(f, 0, 1), facets(f, 1, 1)};

    // Get cell geometry
    auto x_dofs0 = md::submdspan(x_dofmap, cells[0], md::full_extent);
    for (std::size_t i = 0; i < x_dofs0.size(); ++i)
      std::copy_n(&x(x_dofs0[i], 0), 3, std::next(cdofs0.begin(), 3 * i));
    auto x_dofs1 = md::submdspan(x_dofmap, cells[1], md::full_extent);
    for (std::size_t i = 0; i < x_dofs1.size(); ++i)
      std::copy_n(&x(x_dofs1[i], 0), 3, std::next(cdofs1.begin(), 3 * i));

    // Get dof maps for cells and pack
    std::span<const std::int32_t> dmap0_cell0 = dmap0.cell_dofs(cells0[0]);
    std::span<const std::int32_t> dmap0_cell1 = dmap0.cell_dofs(cells0[1]);
    dmapjoint0.resize(dmap0_cell0.size() + dmap0_cell1.size());
    std::ranges::copy(dmap0_cell0, dmapjoint0.begin());
    std::ranges::copy(dmap0_cell1,
                      std::next(dmapjoint0.begin(), dmap0_cell0.size()));

    std::span<const std::int32_t> dmap1_cell0 = dmap1.cell_dofs(cells1[0]);
    std::span<const std::int32_t> dmap1_cell1 = dmap1.cell_dofs(cells1[1]);
    dmapjoint1.resize(dmap1_cell0.size() + dmap1_cell1.size());
    std::ranges::copy(dmap1_cell0, dmapjoint1.begin());
    std::ranges::copy(dmap1_cell1,
                      std::next(dmapjoint1.begin(), dmap1_cell0.size()));

    const int num_rows = bs0 * dmapjoint0.size();
    const int num_cols = bs1 * dmapjoint1.size();

    // Tabulate tensor
    Ae.resize(num_rows * num_cols);
    std::ranges::fill(Ae, 0);

    std::array perm = perms.empty()
                          ? std::array<std::uint8_t, 2>{0, 0}
                          : std::array{perms(cells[0], local_facet[0]),
                                       perms(cells[1], local_facet[1])};
    kernel(Ae.data(), &coeffs(f, 0, 0), constants.data(), cdofs.data(),
           local_facet.data(), perm.data());

    // Local element layout is a 2x2 block matrix with structure
    //
    //   cell0cell0  |  cell0cell1
    //   cell1cell0  |  cell1cell1
    //
    // where each block is element tensor of size (dmap0, dmap1).

    std::span<T> _Ae(Ae);
    std::span<T> sub_Ae0 = _Ae.subspan(bs0 * dmap0_cell0.size() * num_cols,
                                       bs0 * dmap0_cell1.size() * num_cols);

    P0(_Ae, cell_info0, cells0[0], num_cols);
    P0(sub_Ae0, cell_info0, cells0[1], num_cols);
    P1T(_Ae, cell_info1, cells1[0], num_rows);

    for (int row = 0; row < num_rows; ++row)
    {
      // DOFs for dmap1 and cell1 are not stored contiguously in
      // the block matrix, so each row needs a separate span access
      std::span<T> sub_Ae1 = _Ae.subspan(
          row * num_cols + bs1 * dmap1_cell0.size(), bs1 * dmap1_cell1.size());
      P1T(sub_Ae1, cell_info1, cells1[1], 1);
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

/// @brief Assemble (accumulate) into a matrix.
///
/// Rows (bc0) and columns (bc1) with Dirichlet conditions are zeroed.
/// Markers (bc0 and bc1) can be empty if no Dirichlet conditions are
/// applied.
///
/// @tparam T Scalar type.
/// @tparam U Geometry type.
/// @param[in] mat_set Function that accumulates computed entries into a
/// matrix.
/// @param[in] a Bilinear form to assemble.
/// @param[in] x Mesh geometry (coordinates).
/// @param[in] constants Constants that appear in `a`.
/// @param[in] coefficients Coefficients that appear in `a`.
/// @param bc0 Marker for rows with Dirichlet boundary conditions
/// applied.
/// @param bc1 Marker for columns with Dirichlet boundary conditions
/// applied.
template <dolfinx::scalar T, std::floating_point U>
void assemble_matrix(
    la::MatSet<T> auto mat_set, const Form<T, U>& a,
    md::mdspan<const scalar_value_t<T>,
               md::extents<std::size_t, md::dynamic_extent, 3>>
        x,
    std::span<const T> constants,
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

  // TODO: Mixed topology with exterior and interior facet integrals.
  //
  // NOTE: Can't just loop over cell types for interior facet integrals
  // because we have a kernel per combination of comparable cell types,
  // rather than one per cell type. Also, we need the dofmaps for two
  // different cell types at the same time.
  const int num_cell_types = mesh->topology()->cell_types().size();
  for (int cell_type_idx = 0; cell_type_idx < num_cell_types; ++cell_type_idx)
  {
    // Geometry dofmap and data
    mdspan2_t x_dofmap = mesh->geometry().dofmap(cell_type_idx);

    // Get dofmap data
    std::shared_ptr<const fem::DofMap> dofmap0
        = a.function_spaces().at(0)->dofmaps(cell_type_idx);
    std::shared_ptr<const fem::DofMap> dofmap1
        = a.function_spaces().at(1)->dofmaps(cell_type_idx);
    assert(dofmap0);
    assert(dofmap1);
    auto dofs0 = dofmap0->map();
    const int bs0 = dofmap0->bs();
    auto dofs1 = dofmap1->map();
    const int bs1 = dofmap1->bs();

    auto element0 = a.function_spaces().at(0)->elements(cell_type_idx);
    assert(element0);
    auto element1 = a.function_spaces().at(1)->elements(cell_type_idx);
    assert(element1);
    fem::DofTransformKernel<T> auto P0
        = element0->template dof_transformation_fn<T>(doftransform::standard);
    fem::DofTransformKernel<T> auto P1T
        = element1->template dof_transformation_right_fn<T>(
            doftransform::transpose);

    std::span<const std::uint32_t> cell_info0;
    std::span<const std::uint32_t> cell_info1;
    if (element0->needs_dof_transformations()
        or element1->needs_dof_transformations()
        or a.needs_facet_permutations())
    {
      mesh0->topology_mutable()->create_entity_permutations();
      mesh1->topology_mutable()->create_entity_permutations();
      cell_info0 = std::span(mesh0->topology()->get_cell_permutation_info());
      cell_info1 = std::span(mesh1->topology()->get_cell_permutation_info());
    }

    for (int i : a.integral_ids(IntegralType::cell))
    {
      auto fn = a.kernel(IntegralType::cell, i, cell_type_idx);
      assert(fn);
      std::span cells = a.domain(IntegralType::cell, i, cell_type_idx);
      std::span cells0 = a.domain_arg(IntegralType::cell, 0, i, cell_type_idx);
      std::span cells1 = a.domain_arg(IntegralType::cell, 1, i, cell_type_idx);
      auto& [coeffs, cstride] = coefficients.at({IntegralType::cell, i});
      assert(cells.size() * cstride == coeffs.size());
      impl::assemble_cells(mat_set, x_dofmap, x, cells, {dofs0, bs0, cells0},
                           P0, {dofs1, bs1, cells1}, P1T, bc0, bc1, fn,
                           md::mdspan(coeffs.data(), cells.size(), cstride),
                           constants, cell_info0, cell_info1);
    }

    md::mdspan<const std::uint8_t, md::dextents<std::size_t, 2>> perms;
    if (a.needs_facet_permutations())
    {
      mesh::CellType cell_type = mesh->topology()->cell_types()[cell_type_idx];
      int num_facets_per_cell
          = mesh::cell_num_entities(cell_type, mesh->topology()->dim() - 1);
      mesh->topology_mutable()->create_entity_permutations();
      const std::vector<std::uint8_t>& p
          = mesh->topology()->get_facet_permutations();
      perms = md::mdspan(p.data(), p.size() / num_facets_per_cell,
                         num_facets_per_cell);
    }

    for (int i : a.integral_ids(IntegralType::exterior_facet))
    {
      if (num_cell_types > 1)
      {
        throw std::runtime_error("Exterior facet integrals with mixed "
                                 "topology aren't supported yet");
      }

      using mdspanx2_t
          = md::mdspan<const std::int32_t,
                       md::extents<std::size_t, md::dynamic_extent, 2>>;

      auto fn = a.kernel(IntegralType::exterior_facet, i, 0);
      assert(fn);
      auto& [coeffs, cstride]
          = coefficients.at({IntegralType::exterior_facet, i});

      std::span f = a.domain(IntegralType::exterior_facet, i, 0);
      mdspanx2_t facets(f.data(), f.size() / 2, 2);
      std::span f0 = a.domain_arg(IntegralType::exterior_facet, 0, i, 0);
      mdspanx2_t facets0(f0.data(), f0.size() / 2, 2);
      std::span f1 = a.domain_arg(IntegralType::exterior_facet, 1, i, 0);
      mdspanx2_t facets1(f1.data(), f1.size() / 2, 2);
      assert((facets.size() / 2) * cstride == coeffs.size());
      impl::assemble_exterior_facets(
          mat_set, x_dofmap, x, facets, {dofs0, bs0, facets0}, P0,
          {dofs1, bs1, facets1}, P1T, bc0, bc1, fn,
          md::mdspan(coeffs.data(), facets.extent(0), cstride), constants,
          cell_info0, cell_info1, perms);
    }

    for (int i : a.integral_ids(IntegralType::interior_facet))
    {
      if (num_cell_types > 1)
      {
        throw std::runtime_error("Interior facet integrals with mixed "
                                 "topology aren't supported yet");
      }

      using mdspanx22_t
          = md::mdspan<const std::int32_t,
                       md::extents<std::size_t, md::dynamic_extent, 2, 2>>;
      using mdspanx2x_t
          = md::mdspan<const T, md::extents<std::size_t, md::dynamic_extent, 2,
                                            md::dynamic_extent>>;

      auto fn = a.kernel(IntegralType::interior_facet, i, 0);
      assert(fn);
      auto& [coeffs, cstride]
          = coefficients.at({IntegralType::interior_facet, i});

      std::span facets = a.domain(IntegralType::interior_facet, i, 0);
      std::span facets0 = a.domain_arg(IntegralType::interior_facet, 0, i, 0);
      std::span facets1 = a.domain_arg(IntegralType::interior_facet, 1, i, 0);
      assert((facets.size() / 4) * 2 * cstride == coeffs.size());
      impl::assemble_interior_facets(
          mat_set, x_dofmap, x,
          mdspanx22_t(facets.data(), facets.size() / 4, 2, 2),
          {*dofmap0, bs0,
           mdspanx22_t(facets0.data(), facets0.size() / 4, 2, 2)},
          P0,
          {*dofmap1, bs1,
           mdspanx22_t(facets1.data(), facets1.size() / 4, 2, 2)},
          P1T, bc0, bc1, fn,
          mdspanx2x_t(coeffs.data(), facets.size() / 4, 2, cstride), constants,
          cell_info0, cell_info1, perms);
    }
  }
}

} // namespace dolfinx::fem::impl
