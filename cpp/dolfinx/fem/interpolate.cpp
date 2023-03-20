// Copyright (C) 2021 Garth N. Wells, Jørgen S. Dokken, Igor A. Baratta,
// Massimiliano Leoni
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
fem::interpolation_coords(const FiniteElement& element,
                          const mesh::Geometry<double>& geometry,
                          std::span<const std::int32_t> cells)
{
  // Get geometry data and the element coordinate map
  const std::size_t gdim = geometry.dim();
  const graph::AdjacencyList<std::int32_t>& x_dofmap = geometry.dofmap();
  std::span<const double> x_g = geometry.x();
  const CoordinateElement& cmap = geometry.cmap();
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
fem::nmm_interpolation_data_t fem::create_nonmatching_meshes_interpolation_data(
    const mesh::Mesh& mesh0, const FiniteElement& element0,
    const mesh::Mesh& mesh1, std::span<const std::int32_t> cells)
{
  // Collect all the points at which values are needed to define the
  // interpolating function
  const std::vector<double> coords_b
      = interpolation_coords(element0, mesh0.geometry(), cells);

  namespace stdex = std::experimental;
  using cmdspan2_t
      = stdex::mdspan<const double, stdex::dextents<std::size_t, 2>>;
  using mdspan2_t = stdex::mdspan<double, stdex::dextents<std::size_t, 2>>;
  cmdspan2_t coords(coords_b.data(), 3, coords_b.size() / 3);

  // Transpose interpolation coords
  std::vector<double> x(coords.size());
  mdspan2_t _x(x.data(), coords_b.size() / 3, 3);
  for (std::size_t j = 0; j < coords.extent(1); ++j)
    for (std::size_t i = 0; i < 3; ++i)
      _x(j, i) = coords(i, j);

  // Determine ownership of each point
  return geometry::determine_point_ownership(mesh1, x);
}
//-----------------------------------------------------------------------------
fem::nmm_interpolation_data_t
fem::create_nonmatching_meshes_interpolation_data(const mesh::Mesh& mesh0,
                                                  const FiniteElement& element0,
                                                  const mesh::Mesh& mesh1)
{
  int tdim = mesh0.topology().dim();
  auto cell_map = mesh0.topology().index_map(tdim);
  assert(cell_map);
  std::int32_t num_cells = cell_map->size_local() + cell_map->num_ghosts();
  std::vector<std::int32_t> cells(num_cells, 0);
  std::iota(cells.begin(), cells.end(), 0);
  return create_nonmatching_meshes_interpolation_data(mesh0, element0, mesh1,
                                                      cells);
}
//-----------------------------------------------------------------------------
