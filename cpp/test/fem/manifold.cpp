// Copyright (C) 2026 Jørgen S. Dokken
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later
//
// Tests for finite element spaces on manifolds (gdim > tdim). A
// Piola-mapped element is pushed forward with a Jacobian of shape
// (gdim, tdim), so its physical value shape is (gdim,) while the
// reference value shape that Basix tabulates is (tdim,).
//
// See https://github.com/FEniCS/dolfinx/issues/3619.

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_range_equals.hpp>

#include <basix/e-raviart-thomas.h>
#include <basix/finite-element.h>

#include <dolfinx/fem/CoordinateElement.h>
#include <dolfinx/fem/FiniteElement.h>
#include <dolfinx/fem/Function.h>
#include <dolfinx/fem/FunctionSpace.h>
#include <dolfinx/fem/interpolate.h>
#include <dolfinx/fem/utils.h>
#include <dolfinx/mesh/Mesh.h>
#include <dolfinx/mesh/cell_types.h>
#include <dolfinx/mesh/utils.h>

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <numeric>
#include <span>
#include <vector>

using namespace dolfinx;

namespace
{
/// Tangent vectors of the plane z = x, in which every cell of
/// `manifold_mesh` lies.
constexpr std::array<std::array<double, 3>, 2> tangents
    = {{{1.0, 0.0, 1.0}, {0.0, 1.0, 0.0}}};

/// Two triangles embedded in R^3, both lying in the plane z = x.
std::shared_ptr<mesh::Mesh<double>> manifold_mesh()
{
  std::vector<std::int64_t> cells{0, 1, 2, 1, 3, 2};
  std::vector<double> x{0.0, 0.0, 0.0, 1.0, 0.0, 1.0,
                        0.0, 1.0, 0.0, 1.0, 1.0, 1.0};
  fem::CoordinateElement<double> cmap(mesh::CellType::triangle, 1);
  return std::make_shared<mesh::Mesh<double>>(mesh::create_mesh(
      MPI_COMM_SELF, cells, cmap, x, {4, 3}, mesh::GhostMode::none));
}

/// Raviart-Thomas space of the given degree on `mesh`.
std::shared_ptr<fem::FunctionSpace<double>>
rt_space(std::shared_ptr<mesh::Mesh<double>> mesh, int degree)
{
  basix::FiniteElement e = basix::element::create_rt<double>(
      basix::cell::type::triangle, degree,
      basix::element::lagrange_variant::legendre, false);
  return std::make_shared<fem::FunctionSpace<double>>(fem::create_functionspace(
      mesh, std::make_shared<const fem::FiniteElement<double>>(e)));
}
} // namespace

TEST_CASE("Piola element value shape on a manifold", "[manifold]")
{
  auto mesh = manifold_mesh();
  REQUIRE(mesh->geometry().dim() == 3);
  REQUIRE(mesh->topology()->dim() == 2);

  auto V = rt_space(mesh, 1);

  // The push-forward produces one component per physical direction.
  CHECK_THAT(V->element()->value_shape(),
             Catch::Matchers::RangeEquals(std::array<std::size_t, 1>{3}));
  CHECK(V->element()->value_size() == 3);

  // Basix tabulates on the reference cell, where the field has tdim
  // components.
  CHECK_THAT(V->element()->reference_value_shape(),
             Catch::Matchers::RangeEquals(std::array<std::size_t, 1>{2}));
  CHECK(V->element()->reference_value_size() == 2);
}

TEST_CASE("Interpolate a tangential constant on a manifold", "[manifold]")
{
  auto mesh = manifold_mesh();
  auto V = rt_space(mesh, 1);

  // A constant field lying in the plane of every cell. RT of any degree
  // reproduces it exactly.
  std::array<double, 3> c;
  for (std::size_t i = 0; i < 3; ++i)
    c[i] = 0.3 * tangents[0][i] + 0.7 * tangents[1][i];

  std::vector<std::int32_t> cells(mesh->topology()->index_map(2)->size_local());
  std::iota(cells.begin(), cells.end(), 0);

  std::vector<double> x = fem::interpolation_coords<double>(
      *V->element(), mesh->geometry(), cells);
  const std::size_t num_points = x.size() / 3;

  // f has shape (value_size, num_points), value_size == gdim here.
  std::vector<double> f(3 * num_points);
  for (std::size_t i = 0; i < 3; ++i)
    std::fill_n(std::next(f.begin(), i * num_points), num_points, c[i]);

  fem::Function<double> u(V);
  REQUIRE_NOTHROW(fem::interpolate<double>(u, std::span<const double>(f),
                                           {3, num_points}, cells));

  // Evaluate at the centroid of cell 0 and compare with c.
  auto x_dofmap = mesh->geometry().dofmaps().front();
  std::array<double, 3> p{0.0, 0.0, 0.0};
  for (std::size_t i = 0; i < x_dofmap.extent(1); ++i)
    for (std::size_t j = 0; j < 3; ++j)
      p[j] += mesh->geometry().x()[3 * x_dofmap(0, i) + j] / x_dofmap.extent(1);

  std::vector<double> values(3);
  std::vector<std::int32_t> cell0{0};
  u.eval(std::span<const double>(p), {1, 3}, cell0, std::span<double>(values),
         {1, 3}, 1e-8, 10);

  for (std::size_t i = 0; i < 3; ++i)
    CHECK(std::abs(values[i] - c[i]) < 1e-12);
}
