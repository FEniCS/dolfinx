// Copyright (C) 2021 Garth N. Wells, Jørgen S. Dokken, Igor A. Baratta
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "interpolate.h"
#include "FiniteElement.h"
#include "FunctionSpace.h"
#include <dolfinx/mesh/Mesh.h>

using namespace dolfinx;

//-----------------------------------------------------------------------------
std::vector<double>
fem::interpolation_coords(const FiniteElement& element, const mesh::Mesh& mesh,
                          std::span<const std::int32_t> cells)
{
  // Get mesh geometry data and the element coordinate map
  const std::size_t gdim = mesh.geometry().dim();
  const graph::AdjacencyList<std::int32_t>& x_dofmap = mesh.geometry().dofmap();

  std::span<const double> x_g = mesh.geometry().x();
  const CoordinateElement& cmap = mesh.geometry().cmap();
  const std::size_t num_dofs_g = cmap.dim();

  // Get the interpolation points on the reference cells
  const auto [X, Xshape] = element.interpolation_points();

  // Evaluate coordinate element basis at reference points
  namespace stdex = std::experimental;
  using cmdspan4_t
      = stdex::mdspan<const double, stdex::dextents<std::size_t, 4>>;
  std::array<std::size_t, 4> phi_shape = cmap.tabulate_shape(0, Xshape[0]);
  std::vector<double> phi_b(
      std::reduce(phi_shape.begin(), phi_shape.end(), 1, std::multiplies{}));
  cmdspan4_t phi_full(phi_b.data(), phi_shape);
  cmap.tabulate(0, X, Xshape, phi_b);
  auto phi = stdex::submdspan(phi_full, 0, stdex::full_extent,
                              stdex::full_extent, 0);

  // Push reference coordinates (X) forward to the physical coordinates
  // (x) for each cell
  std::vector<double> coordinate_dofs(num_dofs_g * gdim, 0);
  std::vector<double> x(3 * (cells.size() * Xshape[0]), 0);
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
    for (std::size_t p = 0; p < Xshape[0]; ++p)
    {
      for (std::size_t j = 0; j < gdim; ++j)
      {
        double acc = 0;
        for (std::size_t k = 0; k < num_dofs_g; ++k)
          acc += phi(p, k) * coordinate_dofs[k * gdim + j];
        x[j * (cells.size() * Xshape[0]) + c * Xshape[0] + p] = acc;
      }
    }
  }

  return x;
}
//-----------------------------------------------------------------------------
