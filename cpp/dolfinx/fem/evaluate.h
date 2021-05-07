// Copyright (C) 2020 Jack S. Hale
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later
//

#pragma once

#include <dolfinx/common/array2d.h>
#include <dolfinx/fem/utils.h>
#include <dolfinx/mesh/Mesh.h>
#include <vector>
#include <xtl/xspan.hpp>

namespace dolfinx::fem
{

template <typename T>
class Expression;

/// Evaluate a UFC expression.
/// @param[out] values An array to evaluate the expression into
/// @param[in] e The expression to evaluate
/// @param[in] active_cells The cells on which to evaluate the
/// expression
template <typename T>
void eval(array2d<T>& values, const fem::Expression<T>& e,
          const xtl::span<const std::int32_t>& active_cells)
{
  // Extract data from Expression
  auto mesh = e.mesh();
  assert(mesh);

  // Prepare coefficients
  const array2d<T> coeffs = dolfinx::fem::pack_coefficients(e);

  // Prepare constants
  const std::vector<T> constant_values = dolfinx::fem::pack_constants(e);

  const auto& fn = e.get_tabulate_expression();

  // Prepare cell geometry
  const graph::AdjacencyList<std::int32_t>& x_dofmap
      = mesh->geometry().dofmap();
  const fem::CoordinateElement& cmap = mesh->geometry().cmap();

  // Prepate cell permutation info
  mesh->topology_mutable().create_entity_permutations();
  const std::vector<std::uint32_t>& cell_info
      = mesh->topology().get_cell_permutation_info();

  // FIXME: Add proper interface for num coordinate dofs
  const std::size_t num_dofs_g = x_dofmap.num_links(0);
  const xt::xtensor<double, 2>& x_g = mesh->geometry().x();

  // Create data structures used in evaluation
  const std::size_t gdim = mesh->geometry().dim();
  std::vector<double> coordinate_dofs(num_dofs_g * gdim);

  // Iterate over cells and 'assemble' into values
  std::vector<T> values_e(e.num_points() * e.value_size(), 0);
  for (std::size_t c = 0; c < active_cells.size(); ++c)
  {
    const std::int32_t cell = active_cells[c];

    auto x_dofs = x_dofmap.links(c);
    for (std::size_t i = 0; i < x_dofs.size(); ++i)
    {
      std::copy_n(xt::row(x_g, x_dofs[i]).cbegin(), gdim,
                  std::next(coordinate_dofs.begin(), i * gdim));
    }

    auto coeff_cell = coeffs.row(cell);
    std::fill(values_e.begin(), values_e.end(), 0.0);
    fn(values_e.data(), coeff_cell.data(), constant_values.data(),
       coordinate_dofs.data());

    for (std::size_t j = 0; j < values_e.size(); ++j)
      values(c, j) = values_e[j];
  }
}

} // namespace dolfinx::fem
