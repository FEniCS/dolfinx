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
Eigen::Array<double, 3, Eigen::Dynamic, Eigen::RowMajor>
interpolate_coords(const fem::FiniteElement& element, const mesh::Mesh& mesh)
{
  using EigenMatrixRowXd
      = Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

  const int tdim = mesh.topology().dim();
  const int gdim = mesh.geometry().dim();
  auto cell_map = mesh.topology().index_map(tdim);
  assert(cell_map);
  const int num_cells = cell_map->size_local() + cell_map->num_ghosts();

  // Get mesh geometry data and the element coordinate map
  const graph::AdjacencyList<std::int32_t>& x_dofmap = mesh.geometry().dofmap();
  const int num_dofs_g = x_dofmap.num_links(0);
  const EigenMatrixRowXd& x_g = mesh.geometry().x();
  const fem::CoordinateElement& cmap = mesh.geometry().cmap();

  // Get the interpolation points on the reference cells
  const EigenMatrixRowXd X = element.interpolation_points();

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

  return _x;
}
//-----------------------------------------------------------------------------
