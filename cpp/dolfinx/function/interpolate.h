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

/// Interpolate an expression f(x). This interface uses an expression
/// function f that has an in/out argument for the expression values. It
/// is primarily to support C code implementations of the expression,
/// e.g. using Numba. Generally the interface where the expression
/// function is a pure function, i.e. the expression values are the
/// return argument, should be preferred.
/// @param[in,out] u The Function to interpolate into
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
  assert(u.function_space());
  auto mesh = u.function_space()->mesh();
  assert(mesh);
  const int tdim = mesh->topology().dim();

  // Note: the following does not exploit any block structure, e.g. for
  // vector Lagrange, which leads to a lot of redundant evaluations.
  // E.g., for a vector Lagrange element the vector-valued expression is
  // evaluted three times at the some point.

  const int value_size = values.cols();

  // FIXME: Dummy coordinate dofs - should limit the interpolation to
  // Lagrange, in which case we don't need coordinate dofs in
  // FiniteElement::transform_values.
  Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
      coordinate_dofs;

  // FIXME: It would be far more elegant and efficient to avoid the need
  // to loop over cells to set the expansion corfficients. Would be much
  // better if the expansion coefficients could be passed straight into
  // Expresion::eval.

  // Loop over cells
  auto element = u.function_space()->element();
  assert(element);
  const int ndofs = element->space_dimension();
  Eigen::Array<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> values_cell(
      ndofs, value_size);

  auto dofmap = u.function_space()->dofmap();
  assert(dofmap);
  assert(dofmap->element_dof_layout);
  std::vector<T> cell_coefficients(dofmap->element_dof_layout->num_dofs());

  Eigen::Matrix<T, Eigen::Dynamic, 1>& coefficients = u.x()->array();

  auto map = mesh->topology().index_map(tdim);
  assert(map);
  const int num_cells = map->size_local() + map->num_ghosts();
  for (int c = 0; c < num_cells; ++c)
  {
    // Get dofmap for cell
    auto cell_dofs = dofmap->cell_dofs(c);
    for (Eigen::Index i = 0; i < cell_dofs.rows(); ++i)
    {
      for (Eigen::Index j = 0; j < value_size; ++j)
        values_cell(i, j) = values(cell_dofs[i], j);
    }

    // FIXME: For vector-valued Lagrange, this function 'throws away'
    // the redundant expression evaluations. It should really be made
    // not necessary.
    element->transform_values(cell_coefficients.data(), values_cell,
                              coordinate_dofs);

    // Copy into expansion coefficient array
    for (Eigen::Index i = 0; i < cell_dofs.rows(); ++i)
      coefficients[cell_dofs[i]] = cell_coefficients[i];
  }
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

  detail::interpolate_from_any<T>(u, v);
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
  // u.function_space()->interpolate(u.x()->array(), f);
  assert(u.function_space());

  // Evaluate expression at dof points
  const Eigen::Array<double, 3, Eigen::Dynamic, Eigen::RowMajor> x
      = u.function_space()->tabulate_dof_coordinates().transpose();
  const Eigen::Array<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> values
      = f(x);

  const auto element = u.function_space()->element();
  assert(element);
  std::vector<int> vshape(element->value_rank(), 1);
  for (std::size_t i = 0; i < vshape.size(); ++i)
    vshape[i] = element->value_dimension(i);
  const int value_size = std::accumulate(std::begin(vshape), std::end(vshape),
                                         1, std::multiplies<>());

  // Note: pybind11 maps 1D NumPy arrays to column vectors for
  // Eigen::Array<T, Eigen::Dynamic,Eigen::Dynamic, Eigen::RowMajor>
  // types, therefore we need to handle vectors as a special case.
  if (values.cols() == 1 and values.rows() != 1)
  {
    if (values.rows() != x.cols())
    {
      throw std::runtime_error("Number of computed values is not equal to the "
                               "number of evaluation points. (1)");
    }
    detail::interpolate_values<T>(u, values);
  }
  else
  {
    if (values.rows() != value_size)
      throw std::runtime_error("Values shape is incorrect. (2)");
    if (values.cols() != x.cols())
    {
      throw std::runtime_error("Number of computed values is not equal to the "
                               "number of evaluation points. (2)");
    }

    detail::interpolate_values<T>(u, values.transpose());
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
  // u.function_space()->interpolate_c(u.x()->array(), f);
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
