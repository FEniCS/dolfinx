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

namespace dolfinx::function
{

template <typename T>
class Function;

/// Interpolate a Function (on possibly non-matching meshes)
/// @param[in,out] u The function to interpolate into
/// @param[in] v The function to be interpolated
template <typename T>
void interpolate(Function<T>& u, const Function<T>& v);

/// Interpolate an expression
/// @param[in,out] u The function to interpolate into
/// @param[in] f The expression to be interpolated
template <typename T>
void interpolate(
    Function<T>& u,
    const std::function<
        Eigen::Array<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>(
            const Eigen::Ref<const Eigen::Array<double, 3, Eigen::Dynamic,
                                                Eigen::RowMajor>>&)>& f);

/// Interpolate an expression f(x). This interface uses an expression
/// function f that has an in/out argument for the expression values. It
/// is primarily to support C code implementations of the expression,
/// e.g. using Numba. Generally the interface where the expression
/// function is a pure function, i.e. the expression values
/// are the return argument, should be preferred.
/// @param[in,out] u The function to interpolate into
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

// Interpolate data. Fills coefficients using 'values', which are the
// values of an expression at each dof.
template <typename T>
void interpolate_values(
    Function<T>& u,
    const Eigen::Ref<const Eigen::Array<T, Eigen::Dynamic, Eigen::Dynamic,
                                        Eigen::RowMajor>>& values)
{
  Eigen::Matrix<T, Eigen::Dynamic, 1>& coefficients = u.x()->array();
  coefficients = Eigen::Map<const Eigen::Array<T, Eigen::Dynamic, 1>>(
      values.data(), coefficients.rows());
}

template <typename T>
void interpolate_from_any(Function<T>& u, const Function<T>& v)
{
  assert(v.function_space());
  const auto element = u.function_space()->element();
  assert(element);
  if (!v.function_space()->has_element(*element))
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

  Eigen::Matrix<T, Eigen::Dynamic, 1>& expansion_coefficients = u.x()->array();

  // Iterate over mesh and interpolate on each cell
  const auto dofmap_u = u.function_space()->dofmap();
  const Eigen::Matrix<T, Eigen::Dynamic, 1>& v_array = v.x()->array();
  const int num_cells = map->size_local() + map->num_ghosts();
  for (int c = 0; c < num_cells; ++c)
  {
    auto dofs_v = dofmap_v->cell_dofs(c);
    auto cell_dofs = dofmap_u->cell_dofs(c);
    assert(dofs_v.size() == cell_dofs.size());
    for (Eigen::Index i = 0; i < dofs_v.size(); ++i)
      expansion_coefficients[cell_dofs[i]] = v_array[dofs_v[i]];
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
  const auto element = u.function_space()->element();
  const int num_dofs_per_cell = element->space_dimension();
  const auto dofmap = u.function_space()->dofmap();

  if (element->family() == "Mixed")
    throw std::runtime_error("Mixed space interpolation not supported (yet?).");

  // u.function_space()->interpolate(u.x()->array(), f);
  assert(u.function_space());

  // Get mesh
  auto mesh = u.function_space()->mesh();
  assert(mesh);
  const int tdim = mesh->topology().dim();
  const int gdim = mesh->geometry().dim();

  // Get cell geometry
  const graph::AdjacencyList<std::int32_t>& x_dofmap
      = mesh->geometry().dofmap();
  const int num_dofs_g = x_dofmap.num_links(0);
  const Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>&
      x_g
      = mesh->geometry().x();

  // Get index map
  auto map = mesh->topology().index_map(tdim);
  assert(map);
  const int num_cells = map->size_local() + map->num_ghosts();

  // Get coordinate map
  const fem::CoordinateElement& cmap = mesh->geometry().cmap();

  // Get interpolation points on reference
  Eigen::ArrayXXd reference_points = element->interpolation_points();

  // Loop over cells and interpolate on each cell
  Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
      coordinate_dofs(num_dofs_g, gdim);
  Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
      mapped_points(num_dofs_per_cell, gdim);
  Eigen::Array<double, 3, Eigen::Dynamic, Eigen::RowMajor> interpolation_points(
      3, num_dofs_per_cell);

  Eigen::Array<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> values(
      element->value_size(), reference_points.rows());

  Eigen::Array<T, Eigen::Dynamic, 1> coeffs(num_dofs_per_cell);

  Eigen::Array<T, Eigen::Dynamic, 1> interpolation_coeffs(
      u.function_space()->dim());

  const bool needs_permutation_data = element->needs_permutation_data();
  if (needs_permutation_data)
    mesh->topology_mutable().create_entity_permutations();

  const Eigen::Array<std::uint32_t, Eigen::Dynamic, 1>& cell_info
      = needs_permutation_data
            ? mesh->topology().get_cell_permutation_info()
            : Eigen::Array<std::uint32_t, Eigen::Dynamic, 1>(num_cells);

  for (int c = 0; c < num_cells; ++c)
  {
    auto x_dofs = x_dofmap.links(c);
    for (int i = 0; i < num_dofs_g; ++i)
      coordinate_dofs.row(i) = x_g.row(x_dofs[i]).head(gdim);
    cmap.push_forward(mapped_points, reference_points, coordinate_dofs);
    interpolation_points.setZero();
    interpolation_points.block(0, 0, gdim, num_dofs_per_cell)
        = mapped_points.transpose();
    values = f(interpolation_points);
    coeffs = element->interpolate_into_cell(values, cell_info[c]);
    auto dofs = dofmap->cell_dofs(c);
    for (int i = 0; i < dofs.size(); ++i)
      interpolation_coeffs(dofs[i]) = coeffs[i];
  }

  detail::interpolate_values<T>(u, interpolation_coeffs);
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
  // Build list of points at which to evaluate the Expression
  const Eigen::Array<double, Eigen::Dynamic, 3, Eigen::RowMajor> x
      = u.function_space()->tabulate_dof_coordinates();

  // Evaluate expression at points
  const auto element = u.function_space()->element();
  assert(element);
  std::vector<int> vshape(element->value_rank(), 1);
  for (std::size_t i = 0; i < vshape.size(); ++i)
    vshape[i] = element->value_dimension(i);
  const int value_size = std::accumulate(std::begin(vshape), std::end(vshape),
                                         1, std::multiplies<>());
  Eigen::Array<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> values(
      x.rows(), value_size);
  f(values, x);

  detail::interpolate_values<T>(u, values);
}
//----------------------------------------------------------------------------

} // namespace dolfinx::function
