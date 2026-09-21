// Copyright (C) 2024-2026 Paul T. Kühner and Jack S. Hale
//
// This file is part of DOLFINX (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include <catch2/catch_test_macros.hpp>

#include <basix/finite-element.h>

#include <dolfinx/common/MPI.h>
#include <dolfinx/fem/Function.h>
#include <dolfinx/fem/FunctionSpace.h>
#include <dolfinx/fem/utils.h>
#include <dolfinx/mesh/Mesh.h>
#include <dolfinx/mesh/cell_types.h>
#include <dolfinx/mesh/generation.h>

#include <algorithm>
#include <cstdint>
#include <memory>
#include <mpi.h>
#include <span>
#include <vector>

using namespace dolfinx;

TEST_CASE("Create Function Space (mismatch of elements)", "[functionspace]")
{
  auto mesh = std::make_shared<mesh::Mesh<double>>(
      dolfinx::mesh::create_rectangle<double>(
          MPI_COMM_SELF, {{{0, 0}, {1, 1}}}, {1, 1}, mesh::CellType::triangle));

  auto element = basix::create_element<double>(
      basix::element::family::P, basix::cell::type::interval, 1,
      basix::element::lagrange_variant::unset,
      basix::element::dpc_variant::unset, false);

  CHECK_THROWS(fem::create_functionspace<double>(
      mesh, std::make_shared<fem::FiniteElement<double>>(element)));
}

TEST_CASE("Functions share neighbourhood communicators", "[functionspace]")
{
  auto mesh = std::make_shared<mesh::Mesh<double>>(
      dolfinx::mesh::create_rectangle<double>(MPI_COMM_WORLD,
                                              {{{0, 0}, {1, 1}}}, {8, 8},
                                              mesh::CellType::triangle));
  auto element = basix::create_element<double>(
      basix::element::family::P, basix::cell::type::triangle, 1,
      basix::element::lagrange_variant::unset,
      basix::element::dpc_variant::unset, false);
  auto V = std::make_shared<const fem::FunctionSpace<double>>(
      fem::create_functionspace<double>(
          mesh, std::make_shared<fem::FiniteElement<double>>(element)));

  // Creating this many Functions with two communicators each would
  // exhaust MPICH's context IDs. Construction is not collective.
  std::vector<fem::Function<double>> functions;
  for (int i = 0; i < 2048; ++i)
    functions.emplace_back(V);
  for (const fem::Function<double>& f : functions)
    CHECK(f.x()->scatterer() == V->dofmap()->scatterer);

  // Destroy on one rank only, then the remaining Functions must still
  // communicate
  if (dolfinx::MPI::rank(MPI_COMM_WORLD) == 0)
    functions.pop_back();
  fem::Function<double>& u = functions.front();
  const double rank = dolfinx::MPI::rank(MPI_COMM_WORLD);
  std::ranges::fill(u.x()->array(), rank);
  u.x()->scatter_fwd();
  const std::int32_t size_local = u.x()->index_map()->size_local();
  std::span<const int> owners = u.x()->index_map()->owners();
  std::span<const double> x = u.x()->array();
  for (std::size_t i = 0; i < owners.size(); ++i)
    CHECK(x[size_local + i] == owners[i]);
}
