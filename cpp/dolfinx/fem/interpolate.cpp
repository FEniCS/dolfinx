// Copyright (C) 2021 Garth N. Wells, Jørgen S. Dokken, Igor A. Baratta
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "interpolate.h"
#include "FiniteElement.h"
#include "FunctionSpace.h"
#include <dolfinx/mesh/Mesh.h>
#include <xtensor/xbuilder.hpp>

using namespace dolfinx;

//-----------------------------------------------------------------------------
std::vector<double>
fem::interpolation_coords(const fem::FiniteElement& element,
                          const mesh::Mesh& mesh,
                          const xtl::span<const std::int32_t>& cells)
{
  // Get mesh geometry data and the element coordinate map
  const std::size_t gdim = mesh.geometry().dim();
  const graph::AdjacencyList<std::int32_t>& x_dofmap = mesh.geometry().dofmap();

  xtl::span<const double> x_g = mesh.geometry().x();
  const fem::CoordinateElement& cmap = mesh.geometry().cmap();
  const std::size_t num_dofs_g = cmap.dim();

  // Get the interpolation points on the reference cells
  const xt::xtensor<double, 2>& X = element.interpolation_points();
  const xt::xtensor<double, 2> phi
      = xt::view(cmap.tabulate(0, X), 0, xt::all(), xt::all(), 0);

  // Push reference coordinates (X) forward to the physical coordinates
  // (x) for each cell
  std::vector<double> coordinate_dofs(num_dofs_g * gdim, 0);
  std::vector<double> x(3 * (cells.size() * X.shape(0)), 0);
  for (std::size_t c = 0; c < cells.size(); ++c)
  {
    // Get geometry data for current cell
    auto x_dofs = x_dofmap.links(cells[c]);
    for (std::size_t i = 0; i < x_dofs.size(); ++i)
    {
      std::copy_n(std::next(x_g.begin(), 3 * x_dofs[i]), gdim,
                  std::next(coordinate_dofs.begin(), i * gdim));
    }

    // Push forward coordinates (X -> x)
    for (std::size_t p = 0; p < X.shape(0); ++p)
    {
      for (std::size_t j = 0; j < gdim; ++j)
      {
        double acc = 0;
        for (std::size_t k = 0; k < num_dofs_g; ++k)
          acc += phi(p, k) * coordinate_dofs[k * gdim + j];
        x[j * (cells.size() * X.shape(0)) + c * X.shape(0) + p] = acc;
      }
    }
  }

  return x;
}
//-----------------------------------------------------------------------------
