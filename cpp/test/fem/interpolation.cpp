// Copyright (C) 2026 Garth N. Wells
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include <algorithm>
#include <array>
#include <basix/finite-element.h>
#include <catch2/catch_test_macros.hpp>
#include <cstddef>
#include <dolfinx/common/types.h>
#include <dolfinx/fem/FiniteElement.h>
#include <dolfinx/fem/Function.h>
#include <dolfinx/fem/FunctionSpace.h>
#include <dolfinx/fem/functionspace_factory.h>
#include <dolfinx/mesh/Mesh.h>
#include <dolfinx/mesh/cell_types.h>
#include <dolfinx/mesh/generation.h>
#include <memory>
#include <mpi.h>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

using namespace dolfinx;

namespace
{
using x_t
    = md::mdspan<const double, md::extents<std::size_t, 3, md::dynamic_extent>>;

/// Lower-triangle entry (i, j), i >= j, of a symmetric 3x3 tensor field,
/// numbered row by row
std::size_t stored_index(std::size_t i, std::size_t j)
{
  return i * (i + 1) / 2 + j;
}

/// Value of stored component k at point p
double stored_value(std::size_t k, x_t x, std::size_t p)
{
  return (k + 1) * x(0, p) + x(1, p) * x(1, p) - k * x(0, p) * x(1, p);
}
} // namespace

TEST_CASE("Interpolate only the stored components of a symmetric tensor",
          "[interpolation]")
{
  auto mesh = std::make_shared<mesh::Mesh<double>>(
      mesh::create_rectangle<double>(MPI_COMM_WORLD, {{{0, 0}, {1, 1}}}, {3, 3},
                                     mesh::CellType::triangle));
  basix::FiniteElement<double> e = basix::create_element<double>(
      basix::element::family::P, basix::cell::type::triangle, 2,
      basix::element::lagrange_variant::gll_warped,
      basix::element::dpc_variant::unset, false);
  auto V = std::make_shared<fem::FunctionSpace<double>>(
      fem::create_functionspace<double>(
          mesh, std::make_shared<fem::FiniteElement<double>>(
                    e, mesh->geometry().dim(), std::vector<std::size_t>{3, 3},
                    true)));

  // All 9 components, row-major
  fem::Function<double> u_full(V);
  u_full.interpolate(
      [](x_t x) -> std::pair<std::vector<double>, std::vector<std::size_t>>
      {
        std::vector<double> f(9 * x.extent(1));
        for (std::size_t i = 0; i < 3; ++i)
          for (std::size_t j = 0; j < 3; ++j)
            for (std::size_t p = 0; p < x.extent(1); ++p)
            {
              f[(3 * i + j) * x.extent(1) + p] = stored_value(
                  stored_index(std::max(i, j), std::min(i, j)), x, p);
            }
        return {std::move(f), {9, x.extent(1)}};
      });

  // Only the 6 stored components
  auto stored = [](x_t x, std::size_t num_components)
      -> std::pair<std::vector<double>, std::vector<std::size_t>>
  {
    std::vector<double> f(num_components * x.extent(1));
    for (std::size_t k = 0; k < num_components; ++k)
      for (std::size_t p = 0; p < x.extent(1); ++p)
        f[k * x.extent(1) + p] = stored_value(k, x, p);
    return {std::move(f), {num_components, x.extent(1)}};
  };
  fem::Function<double> u_stored(V);
  u_stored.interpolate([&stored](x_t x) { return stored(x, 6); });

  std::span<const double> full = u_full.x()->array();
  CHECK(std::ranges::any_of(full, [](double v) { return v != 0; }));
  CHECK(std::ranges::equal(u_stored.x()->array(), full));

  // Any other number of components is rejected
  CHECK_THROWS_AS(
      u_stored.interpolate([&stored](x_t x) { return stored(x, 5); }),
      std::invalid_argument);
}
