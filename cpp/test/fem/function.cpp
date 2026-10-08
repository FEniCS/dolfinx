// Copyright (C) 2026 Jack S. Hale
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later
//
// Unit tests for fem::Function

#include <basix/finite-element.h>
#include <catch2/catch_test_macros.hpp>
#include <cstddef>
#include <dolfinx/common/IndexMap.h>
#include <dolfinx/common/Scatterer.h>
#include <dolfinx/fem/Function.h>
#include <dolfinx/fem/FunctionSpace.h>
#include <dolfinx/fem/utils.h>
#include <dolfinx/graph/partition.h>
#include <dolfinx/la/Vector.h>
#include <dolfinx/mesh/Mesh.h>
#include <dolfinx/mesh/generation.h>
#include <memory>
#include <mpi.h>
#include <stdexcept>
#include <vector>

using namespace dolfinx;

TEST_CASE("Function from vector with a shared scatterer", "[function]")
{
  auto mesh = std::make_shared<mesh::Mesh<double>>(mesh::create_box<double>(
      MPI_COMM_WORLD, {{{0.0, 0.0, 0.0}, {1.0, 1.0, 1.0}}}, {3, 3, 3},
      mesh::CellType::tetrahedron, graph::partition_graph));
  auto element = basix::create_element<double>(
      basix::element::family::P, basix::cell::type::tetrahedron, 1,
      basix::element::lagrange_variant::unset,
      basix::element::dpc_variant::unset, false);
  auto V = std::make_shared<fem::FunctionSpace<double>>(
      fem::create_functionspace<double>(
          mesh, std::make_shared<fem::FiniteElement<double>>(
                    element, std::vector<std::size_t>{3})));

  std::shared_ptr<const common::IndexMap> map = V->dofmap()->index_map;
  int bs = V->dofmap()->index_map_bs();
  REQUIRE(bs == 3);

  // Function u1 built from a vector sharing the scatterer of u0
  fem::Function<double> u0(V);
  auto x = std::make_shared<la::Vector<double>>(map, bs, u0.x()->scatterer());
  fem::Function<double> u1(V, x);
  CHECK(u1.x() == x);
  CHECK(u1.x()->scatterer() == u0.x()->scatterer());

  // Sub-functions share the parent vector and scatterer
  CHECK(u1.sub(0).x()->scatterer() == u0.x()->scatterer());

  // Wrong block size
  auto x_bs = std::make_shared<la::Vector<double>>(map, 1, u0.x()->scatterer());
  CHECK_THROWS_AS(fem::Function<double>(V, x_bs), std::invalid_argument);

  // Wrong index map (no ghosts)
  auto map_local
      = std::make_shared<common::IndexMap>(MPI_COMM_WORLD, map->size_local());
  auto x_map = std::make_shared<la::Vector<double>>(map_local, bs);
  if (map->num_ghosts() > 0)
    CHECK_THROWS_AS(fem::Function<double>(V, x_map), std::invalid_argument);
}
