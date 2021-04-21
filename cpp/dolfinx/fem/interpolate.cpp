// Copyright (C) 2021 Garth N. Wells, Jørgen S. Dokken, Igor A. Baratta
//
// This file is part of DOLFINX (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "interpolate.h"
#include "FiniteElement.h"
#include <dolfinx/mesh/Mesh.h>
#include <xtensor/xbuilder.hpp>
#include <xtensor/xtensor.hpp>
#include <xtensor/xview.hpp>

using namespace dolfinx;

//-----------------------------------------------------------------------------
xt::xtensor<double, 2>
fem::interpolation_coords(const fem::FiniteElement& element,
                          const mesh::Mesh& mesh,
                          const xtl::span<const std::int32_t>& cells)
{
  // Get mesh geometry data and the element coordinate map
  const std::size_t gdim = mesh.geometry().dim();
  const graph::AdjacencyList<std::int32_t>& x_dofmap = mesh.geometry().dofmap();
  const std::size_t num_dofs_g = x_dofmap.num_links(0);
  const xt::xtensor<double, 2>& x_g = mesh.geometry().x();

  const fem::CoordinateElement& cmap = mesh.geometry().cmap();

  // Get the interpolation points on the reference cells
  const xt::xtensor<double, 2>& X
      = element.interpolation_points();
  const xt::xtensor<double, 2> phi
      = xt::view(cmap.tabulate(0, X), 0, xt::all(), xt::all(), 0);
  // std::cout << "P-------" << std::endl;
  // std::cout << phi << std::endl;
  // std::cout << "X-------" << std::endl;
  // std::cout << X << std::endl;
  // std::cout << "--------" << std::endl;

  // Push reference coordinates (X) forward to the physical coordinates
  // (x) for each cell
  xt::xtensor<double, 2> x_cell = xt::zeros<double>({X.shape(0), gdim});
  xt::xtensor<double, 2> coordinate_dofs
      = xt::zeros<double>({num_dofs_g, gdim});
  std::array<std::size_t, 2> shape = {3, cells.size() * X.shape(0)};
  xt::xtensor<double, 2> x = xt::zeros<double>(shape);
  for (std::size_t c = 0; c < cells.size(); ++c)
  {
    // Get geometry data for current cell
    auto x_dofs = x_dofmap.links(cells[c]);
    for (std::size_t i = 0; i < x_dofs.size(); ++i)
    {
      std::copy_n(xt::row(x_g, x_dofs[i]).begin(), gdim,
                  std::next(coordinate_dofs.begin(), i * gdim));
    }

    // Push forward coordinates (X -> x)
    cmap.push_forward(x_cell, coordinate_dofs, phi);
    for (std::size_t i = 0; i < x_cell.shape(0); ++i)
      for (std::size_t j = 0; j < x_cell.shape(1); ++j)
        x(j, c * X.shape(0) + i) = x_cell(i, j);
  }

  return x;
}
//-----------------------------------------------------------------------------
