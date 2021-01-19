// Copyright (C) 2020 Garth N. Wells
//
// This file is part of DOLFINX (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include "Function.h"
#include "FunctionSpace.h"
#include <Eigen/Dense>
#include <dolfinx/fem/DofMap.h>
#include <dolfinx/fem/FiniteElement.h>
#include <dolfinx/mesh/Mesh.h>
#include <functional>

namespace dolfinx::fem
{

template <typename T>
class Function;

/// Interpolate a finite element Function (on possibly non-matching
/// meshes) in another finite element space
/// @param[out] u The function to interpolate into
/// @param[in] v The function to be interpolated
template <typename T>
void interpolate(Function<T>& u, const Function<T>& v);

/// Interpolate an expression in a finite element space
/// @param[out] u The function to interpolate into
/// @param[in] f The expression to be interpolated
template <typename T>
void interpolate(
    Function<T>& u,
    const std::function<
        Eigen::Array<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>(
            const Eigen::Ref<const Eigen::Array<double, 3, Eigen::Dynamic,
                                                Eigen::RowMajor>>&)>& f);

/// Interpolate an expression f(x)
///
/// @note  This interface uses an expression function f that has an
/// in/out argument for the expression values. It is primarily to
/// support C code implementations of the expression, e.g. using Numba.
/// Generally the interface where the expression function is a pure
/// function, i.e. the expression values are the return argument, should
/// be preferred.
///
/// @param[out] u The function to interpolate into
/// @param[in] f The expression to be interpolated
template <typename T>
void interpolate_c(
    Function<T>& u,
    const std::function<void(
        Eigen::Ref<
            Eigen::Array<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>,
        const Eigen::Ref<const Eigen::Array<double, Eigen::Dynamic, 3,
                                            Eigen::RowMajor>>&)>& f);

namespace detail
{

template <typename T>
void interpolate_from_any(Function<T>& u, const Function<T>& v)
{
  assert(v.function_space());
  const auto element = u.function_space()->element();
  assert(element);
  if (v.function_space()->element()->hash() != element->hash())
  {
    throw std::runtime_error("Restricting finite elements function in "
                             "different elements not supported.");
  }

  const auto mesh = u.function_space()->mesh();
  assert(mesh);
  assert(v.function_space()->mesh());
  if (mesh->id() != v.function_space()->mesh()->id())
  {
    throw std::runtime_error(
        "Interpolation on different meshes not supported (yet).");
  }
  const int tdim = mesh->topology().dim();

  // Get dofmaps
  assert(v.function_space());
  std::shared_ptr<const fem::DofMap> dofmap_v = v.function_space()->dofmap();
  assert(dofmap_v);
  auto map = mesh->topology().index_map(tdim);
  assert(map);

  std::vector<T>& coeffs = u.x()->mutable_array();

  // Iterate over mesh and interpolate on each cell
  const auto dofmap_u = u.function_space()->dofmap();
  const std::vector<T>& v_array = v.x()->array();
  const int num_cells = map->size_local() + map->num_ghosts();
  const int bs = dofmap_v->bs();
  assert(bs == dofmap_u->bs());
  for (int c = 0; c < num_cells; ++c)
  {
    tcb::span<const std::int32_t> dofs_v = dofmap_v->cell_dofs(c);
    tcb::span<const std::int32_t> cell_dofs = dofmap_u->cell_dofs(c);
    assert(dofs_v.size() == cell_dofs.size());
    for (std::size_t i = 0; i < dofs_v.size(); ++i)
    {
      for (int k = 0; k < bs; ++k)
        coeffs[bs * cell_dofs[i] + k] = v_array[bs * dofs_v[i] + k];
    }
  }
}

} // namespace detail

//----------------------------------------------------------------------------
template <typename T>
void interpolate(Function<T>& u, const Function<T>& v)
{
  assert(u.function_space());
  const auto element = u.function_space()->element();
  assert(element);

  // Check that function ranks match
  if (int rank_v = v.function_space()->element()->value_rank();
      element->value_rank() != rank_v)
  {
    throw std::runtime_error("Cannot interpolate function into function space. "
                             "Rank of function ("
                             + std::to_string(rank_v)
                             + ") does not match rank of function space ("
                             + std::to_string(element->value_rank()) + ")");
  }

  // Check that function dimension match
  for (int i = 0; i < element->value_rank(); ++i)
  {
    if (int v_dim = v.function_space()->element()->value_dimension(i);
        element->value_dimension(i) != v_dim)
    {
      throw std::runtime_error(
          "Cannot interpolate function into function space. "
          "Dimension "
          + std::to_string(i) + " of function (" + std::to_string(v_dim)
          + ") does not match dimension " + std::to_string(i)
          + " of function space(" + std::to_string(element->value_dimension(i))
          + ")");
    }
  }

  detail::interpolate_from_any(u, v);
}
//----------------------------------------------------------------------------
template <typename T>
void interpolate(
    Function<T>& u,
    const std::function<
        Eigen::Array<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>(
            const Eigen::Ref<const Eigen::Array<double, 3, Eigen::Dynamic,
                                                Eigen::RowMajor>>&)>& f)
{
  using EigenMatrixRowXd
      = Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

  const auto element = u.function_space()->element();
  assert(element);
  const int element_bs = element->block_size();

  if (int num_sub = element->num_sub_elements();
      num_sub > 0 and num_sub != element_bs)
  {
    throw std::runtime_error("Cannot directly interpolate a mixed space. "
                             "Interpolate into subspaces.");
  }

  // Get mesh
  assert(u.function_space());
  auto mesh = u.function_space()->mesh();
  assert(mesh);
  const int tdim = mesh->topology().dim();
  const int gdim = mesh->geometry().dim();
  auto cell_map = mesh->topology().index_map(tdim);
  assert(cell_map);
  const int num_cells = cell_map->size_local() + cell_map->num_ghosts();

  // Get mesh geometry data and the element coordinate map
  const graph::AdjacencyList<std::int32_t>& x_dofmap
      = mesh->geometry().dofmap();
  const int num_dofs_g = x_dofmap.num_links(0);
  const EigenMatrixRowXd& x_g = mesh->geometry().x();
  const fem::CoordinateElement& cmap = mesh->geometry().cmap();

  // Get the interpolation points on the reference cells
  const Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>& X
      = element->interpolation_points();

  const bool needs_permutation_data = element->needs_permutation_data();
  if (needs_permutation_data)
    mesh->topology_mutable().create_entity_permutations();
  const std::vector<std::uint32_t>& cell_info
      = needs_permutation_data ? mesh->topology().get_cell_permutation_info()
                               : std::vector<std::uint32_t>(num_cells);

  // Push reference coordinates (X) forward to the physical coordinates
  // (x) for each cell
  EigenMatrixRowXd x_cell(X.rows(), gdim);
  std::vector<double> x;
  EigenMatrixRowXd coordinate_dofs(num_dofs_g, gdim);
  for (int c = 0; c < num_cells; ++c)
  {
    // Get geometry data for current cell
    auto x_dofs = x_dofmap.links(c);
    for (int i = 0; i < num_dofs_g; ++i)
      coordinate_dofs.row(i) = x_g.row(x_dofs[i]).head(gdim);

    // Push forward coordinates (X -> x)
    cmap.push_forward(x_cell, X, coordinate_dofs);
    x.insert(x.end(), x_cell.data(), x_cell.data() + x_cell.size());
  }

  // Re-pack points (each row for a given coordinate component) and pad
  // up to gdim with zero
  Eigen::Array<double, 3, Eigen::Dynamic, Eigen::RowMajor> _x
      = Eigen::Array<double, 3, Eigen::Dynamic, Eigen::RowMajor>::Zero(
          3, x.size() / gdim);
  for (int i = 0; i < gdim; ++i)
  {
    _x.row(i)
        = Eigen::Map<Eigen::ArrayXd, 0, Eigen::InnerStride<Eigen::Dynamic>>(
            x.data() + i, x.size() / gdim,
            Eigen::InnerStride<Eigen::Dynamic>(gdim));
  }

  // Evaluate function at physical points. The returned array has a
  // number of rows equal to the number of components of the function,
  // and the number of columns is equal to the number of evaluation
  // points. Scalar case needs special handling as pybind11 will return
  // a column array when we need a row array.
  Eigen::Array<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> values;
  values = f(_x);

  // FIXME: This is hack for NumPy/pybind11/Eigen that returns 1D arrays a
  // column vectors. Fix in the pybind11 layer?
  if (element->value_size() == 1 and values.rows() > 1)
    values = values.transpose().eval();

  // Get dofmap
  const auto dofmap = u.function_space()->dofmap();
  assert(dofmap);
  const int dofmap_bs = dofmap->bs();

  // NOTE: The below loop over cells could be skipped for some elements,
  // e.g. Lagrange, where the interpolation is just the identity.

  // Loop over cells and compute interpolation dofs
  const int num_scalar_dofs = element->space_dimension() / element_bs;
  const int value_size = element->value_size() / element_bs;
  std::vector<T>& coeffs = u.x()->mutable_array();
  Eigen::Array<T, Eigen::Dynamic, 1> _coeffs(num_scalar_dofs);
  Eigen::Array<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> _vals;
  for (int c = 0; c < num_cells; ++c)
  {
    auto dofs = dofmap->cell_dofs(c);
    for (int k = 0; k < element_bs; ++k)
    {
      // Extract computed expression values for element block k
      _vals = values.block(k, c * X.rows(), value_size, X.rows());

      // Get element degrees of freedom for block
      element->interpolate(_vals, cell_info[c], _coeffs);
      assert(_coeffs.size() == num_scalar_dofs);

      // Copy interpolation dofs into coefficient vector
      for (int i = 0; i < num_scalar_dofs; ++i)
      {
        const int dof = i * element_bs + k;
        std::div_t pos = std::div(dof, dofmap_bs);
        coeffs[dofmap_bs * dofs[pos.quot] + pos.rem] = _coeffs[i];
      }
    }
  }
}
//----------------------------------------------------------------------------
template <typename T>
void interpolate_c(
    Function<T>& u,
    const std::function<void(
        Eigen::Ref<
            Eigen::Array<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>,
        const Eigen::Ref<const Eigen::Array<double, Eigen::Dynamic, 3,
                                            Eigen::RowMajor>>&)>& f)
{
  const auto element = u.function_space()->element();
  assert(element);
  std::vector<int> vshape(element->value_rank(), 1);
  for (std::size_t i = 0; i < vshape.size(); ++i)
    vshape[i] = element->value_dimension(i);
  const int value_size = std::accumulate(std::begin(vshape), std::end(vshape),
                                         1, std::multiplies<>());

  auto fn =
      [value_size,
       &f](const Eigen::Ref<
           const Eigen::Array<double, 3, Eigen::Dynamic, Eigen::RowMajor>>& x) {
        Eigen::Array<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> values(
            x.rows(), value_size);
        f(values, x);
        return values;
      };

  interpolate<T>(u, fn);
}
//----------------------------------------------------------------------------

} // namespace dolfinx::fem
