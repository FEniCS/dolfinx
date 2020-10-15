// Copyright (C) 2020 Jack S. Hale
//
// This file is part of DOLFINX (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later
//

#pragma once

#include <vector>

#include <Eigen/Dense>
#include <dolfinx/fem/utils.h>
#include <dolfinx/mesh/Mesh.h>

namespace dolfinx::function
{

template <typename T>
class Expression;
/// Evaluate a UFC expression.
/// @param[in,out] values An array to evaluate the expression into
/// @param[in] e The expression to evaluate
/// @param[in] active_cells The cells on which to evaluate the expression
template <typename T>
void eval(
    Eigen::Ref<Eigen::Array<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
        values,
    const function::Expression<T>& e,
    const std::vector<std::int32_t>& active_cells)
{
  // Extract data from Expression
  auto mesh = e.mesh();
  assert(mesh);

  // Prepare coefficients
  Eigen::Array<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> coeffs
      = dolfinx::fem::pack_coefficients<T, function::Expression<T>>(e);

  // Prepare constants
  if (!e.all_constants_set())
    throw std::runtime_error("Unset constant in Form");
  const Eigen::Array<T, Eigen::Dynamic, 1> constant_values
      = dolfinx::fem::pack_constants<T, function::Expression<T>>(e);

  const auto& fn = e.get_tabulate_expression();

  // Prepare cell geometry
  const graph::AdjacencyList<std::int32_t>& x_dofmap
      = mesh->geometry().dofmap();

  // FIXME: Add proper interface for num coordinate dofs
  const int num_dofs_g = x_dofmap.num_links(0);
  const Eigen::Array<double, Eigen::Dynamic, 3, Eigen::RowMajor>& x_g
      = mesh->geometry().x();

  // Create data structures used in evaluation
  const int gdim = mesh->geometry().dim();
  Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
      coordinate_dofs(num_dofs_g, gdim);

  Eigen::Array<T, Eigen::Dynamic, 1> values_e;
  const Eigen::Index num_points = e.num_points();
  const Eigen::Index value_size = e.value_size();
  const Eigen::Index size = num_points * value_size;
  values_e.setZero(size);

  // Iterate over cells and 'assemble' into values
  Eigen::Index i = 0;
  for (std::int32_t c : active_cells)
  {
    auto x_dofs = x_dofmap.links(c);
    for (Eigen::Index j = 0; j < num_dofs_g; ++j)
    {
      const auto x_dof = x_dofs[j];
      for (Eigen::Index k = 0; k < gdim; ++k)
        coordinate_dofs(j, k) = x_g(x_dof, k);
    }

    auto coeff_cell = coeffs.row(c);

    // Experimentally faster than .setZero().
    for (Eigen::Index j = 0; j < size; j++)
      values_e(j) = 0.0;

    fn(values_e.data(), coeff_cell.data(), constant_values.data(),
       coordinate_dofs.data());

    values.row(i) = values_e;
    ++i;
  }
}

} // namespace dolfinx::function
