// Copyright (C) 2021 Garth N. Wells
//
// This file is part of DOLFINX (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "interpolate.h"
#include "FiniteElement.h"
#include <dolfinx/mesh/Mesh.h>

using namespace dolfinx;

//-----------------------------------------------------------------------------
array2d<double>
fem::interpolation_coords(const fem::FiniteElement& element,
                          const mesh::Mesh& mesh,
                          const tcb::span<const std::int32_t>& cells)
{
  // Get mesh geometry data and the element coordinate map
  const int gdim = mesh.geometry().dim();
  const graph::AdjacencyList<std::int32_t>& x_dofmap = mesh.geometry().dofmap();
  const int num_dofs_g = x_dofmap.num_links(0);
  const array2d<double>& x_g = mesh.geometry().x();

  const fem::CoordinateElement& cmap = mesh.geometry().cmap();

  // Get the interpolation points on the reference cells
  const array2d<double> X = element.interpolation_points();

  // Push reference coordinates (X) forward to the physical coordinates
  // (x) for each cell
  array2d<double> x_cell(X.shape[0], gdim);
  array2d<double> x(3, cells.size() * X.shape[0], 0.0);
  array2d<double> coordinate_dofs(num_dofs_g, gdim);
  for (std::size_t c = 0; c < cells.size(); ++c)
  {
    // Get geometry data for current cell
    auto x_dofs = x_dofmap.links(cells[c]);
    for (int i = 0; i < num_dofs_g; ++i)
      for (int j = 0; j < gdim; ++j)
        coordinate_dofs(i, j) = x_g(x_dofs[i], j);

    // Push forward coordinates (X -> x)
    cmap.push_forward(x_cell, X, coordinate_dofs);
    for (std::size_t i = 0; i < x_cell.shape[0]; ++i)
      for (std::size_t j = 0; j < x_cell.shape[1]; ++j)
        x(j, c * X.shape[0] + i) = x_cell(i, j);
  }

  return x;
}
//-----------------------------------------------------------------------------
