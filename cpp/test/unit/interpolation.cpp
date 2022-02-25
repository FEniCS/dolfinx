// Copyright (C) 2022 Massimiliano Leoni
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include <catch2/catch.hpp>
#include <dolfinx/fem/FunctionSpace.h>
#include <dolfinx/fem/utils.h>
#include <dolfinx/la/petsc.h>
#include <dolfinx/mesh/Mesh.h>
#include <dolfinx/mesh/generation.h>
#include <mpi.h>

#include "interpolation.h"

using namespace dolfinx;

namespace
{
void test_interpolation_different_meshes()
{
  const std::array<std::size_t, 3> subdivisions = {5, 5, 5};

  auto meshL = std::make_shared<mesh::Mesh>(mesh::create_box(
      MPI_COMM_WORLD, {{{0, 0, 0}, {1, 1, 1}}}, subdivisions,
      mesh::CellType::tetrahedron, mesh::GhostMode::shared_facet));

  auto meshR = std::make_shared<mesh::Mesh>(mesh::create_box(
      MPI_COMM_WORLD, {{{0, 0, 0}, {1, 1, 1}}}, subdivisions,
      mesh::CellType::tetrahedron, mesh::GhostMode::shared_facet));

  auto VL = std::make_shared<fem::FunctionSpace>(fem::create_functionspace(
      functionspace_form_interpolation_a, "u", meshL));
  auto VR = std::make_shared<fem::FunctionSpace>(fem::create_functionspace(
      functionspace_form_interpolation_a, "u", meshR));

  auto uL = std::make_shared<fem::Function<PetscScalar>>(VL);
  auto uR = std::make_shared<fem::Function<PetscScalar>>(VR);

  uL->interpolate(
      [](auto& x)
      {
        auto r = xt::zeros_like(x);
        for (std::size_t i = 0; i < x.shape(1); ++i)
        {
          r(0, i) = std::cos(10 * x(0, i)) * std::sin(10 * x(2, i));
          r(1, i) = std::sin(10 * x(0, i)) * std::sin(10 * x(2, i));
          r(2, i) = std::cos(10 * x(0, i)) * std::cos(10 * x(2, i));
        }
        return r;
      });

  uR->interpolate(*uL);

  auto uR_ex = std::make_shared<fem::Function<PetscScalar>>(VR);
  uR_ex->interpolate(
      [](auto& x)
      {
        auto r = xt::zeros_like(x);
        for (std::size_t i = 0; i < x.shape(1); ++i)
        {
          r(0, i) = std::cos(10 * x(0, i)) * std::sin(10 * x(2, i));
          r(1, i) = std::sin(10 * x(0, i)) * std::sin(10 * x(2, i));
          r(2, i) = std::cos(10 * x(0, i)) * std::cos(10 * x(2, i));
        }
        return r;
      });

  const PetscReal diffNorm = std::sqrt(std::transform_reduce(
      uR->x()->array().cbegin(), uR->x()->array().cend(),
      uR_ex->x()->array().cbegin(), static_cast<PetscReal>(0), std::plus<>(),
      [](const auto& a, const auto& b)
      { return std::real((a - b) * std::conj(a - b)); }));

  if (diffNorm > 1e-13)
  {
    throw std::runtime_error("Interpolation on different meshes failed.");
  }
}

} // namespace

TEST_CASE("Interpolation between different meshes",
          "[interpolation_different_meshes]")
{
  CHECK_NOTHROW(test_interpolation_different_meshes());
}
