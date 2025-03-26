// Copyright (C) 2024 Paul T. Kühner
//
// This file is part of DOLFINX (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include <concepts>
#include <cstdint>
#include <memory>
#include <optional>
#include <vector>

#include <mpi.h>

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <catch2/matchers/catch_matchers_range_equals.hpp>

#include <dolfinx/common/IndexMap.h>
#include <dolfinx/common/MPI.h>
#include <dolfinx/fem/FunctionSpace.h>
#include <dolfinx/fem/utils.h>
#include <dolfinx/la/MatrixCSR.h>
#include <dolfinx/la/SparsityPattern.h>
#include <dolfinx/mesh/cell_types.h>
#include <dolfinx/mesh/generation.h>
#include <dolfinx/mesh/utils.h>
#include <dolfinx/refinement/plaza.h>
#include <dolfinx/refinement/refine.h>
#include <dolfinx/transfer/transfer_matrix.h>

using namespace dolfinx;
using namespace Catch::Matchers;

namespace
{
template <typename T>
constexpr auto EPS = std::numeric_limits<T>::epsilon();
} // namespace

template <std::floating_point T>
constexpr auto weight = [](std::int32_t d) -> T { return d == 0 ? 1. : .5; };

TEMPLATE_TEST_CASE("Transfer Matrix 1D", "[transfer_matrix]",
                   double) // TODO: float
{
  using T = TestType;

  auto mesh_coarse
      = std::make_shared<mesh::Mesh<T>>(dolfinx::mesh::create_interval<T>(
          MPI_COMM_SELF, 2, {0.0, 1.0}, mesh::GhostMode::none));

  auto [mesh_fine, parent_cell, parent_facet]
      = refinement::refine(*mesh_coarse, std::nullopt);

  auto element = basix::create_element<T>(
      basix::element::family::P, basix::cell::type::interval, 1,
      basix::element::lagrange_variant::unset,
      basix::element::dpc_variant::unset, false);

  auto V_coarse = std::make_shared<fem::FunctionSpace<T>>(
      fem::create_functionspace<T>(mesh_coarse, element, {}));
  auto V_fine
      = std::make_shared<fem::FunctionSpace<T>>(fem::create_functionspace<T>(
          std::make_shared<mesh::Mesh<T>>(mesh_fine), element, {}));

  mesh_fine.topology()->create_connectivity(1, 0);
  mesh_fine.topology()->create_connectivity(0, 1);

  std::vector<std::int64_t> inclusion_map
      = transfer::inclusion_mapping(*mesh_coarse, mesh_fine);

  CHECK_THAT(inclusion_map, RangeEquals(std::vector<std::int64_t>{0, 2, 4}));

  la::SparsityPattern sp
      = transfer::create_sparsity_pattern(*V_coarse, *V_fine, inclusion_map);

  la::MatrixCSR<T> transfer_matrix(std::move(sp), la::BlockMode::compact);
  transfer::assemble_transfer_matrix<T>(transfer_matrix.mat_set_values(),
                                        *V_coarse, *V_fine, inclusion_map,
                                        weight<T>);

  std::vector<T> expected{1.0, .5, 0, 0, 0, 0, .5, 1, .5, 0, 0, 0, 0, .5, 1};
  CHECK_THAT(transfer_matrix.to_dense(),
             RangeEquals(expected, [](auto a, auto b)
                         { return std::abs(a - b) <= EPS<T>; }));
}

TEMPLATE_TEST_CASE("Transfer Matrix 1D (parallel)", "[transfer_matrix]",
                   double) // TODO: float
{
  using T = TestType;

  int comm_size = dolfinx::MPI::size(MPI_COMM_WORLD);
  int rank = dolfinx::MPI::rank(MPI_COMM_WORLD);

  // TODO: see https://github.com/FEniCS/dolfinx/issues/3358
  //   auto part = mesh::create_cell_partitioner(mesh::GhostMode::shared_facet);
  mesh::CellPartitionFunction part
      = [&](MPI_Comm /* comm */, int /* nparts */,
            const std::vector<mesh::CellType>& /* cell_types */,
            const std::vector<std::span<const std::int64_t>>& /* cells */)
  {
    std::vector<std::vector<std::int32_t>> data;
    if (comm_size == 1)
      data = {{0}, {0}, {0}, {0}};
    else if (comm_size == 2)
      data = {{0}, {0}, {0}, {0}, {0, 1}, {1, 0}, {1}, {1}, {1}};
    else if (comm_size == 3)
      data = {{1}, {1},    {1},    {1}, {1, 2}, {2, 1}, {2},
              {2}, {2, 0}, {0, 2}, {0}, {0},    {0},    {0}};
    else
      FAIL("Test only supports <= 3 processes");
    return graph::AdjacencyList<std::int32_t>(std::move(data));
  };

  auto mesh_coarse = std::make_shared<mesh::Mesh<T>>(
      mesh::create_interval<T>(MPI_COMM_WORLD, 5 * comm_size - 1, {0., 1.},
                               mesh::GhostMode::shared_facet, part));

  // TODO: see https://github.com/FEniCS/dolfinx/issues/3358
  //   auto part_refine
  //       = mesh::create_cell_partitioner(mesh::GhostMode::shared_facet);
  mesh::CellPartitionFunction part_refine
      = [&](MPI_Comm comm, int /* nparts */,
            const std::vector<mesh::CellType>& /* cell_types */,
            const std::vector<std::span<const std::int64_t>>& /* cells */)
  {
    std::vector<std::vector<std::int32_t>> data;
    if (dolfinx::MPI::size(comm) == 1)
      data = {{0}, {0}, {0}, {0}, {0}, {0}, {0}, {0}};
    else if (dolfinx::MPI::size(comm) == 2)
    {
      if (rank == 0)
        data = {{0}, {0}, {0}, {0}, {0}, {0}, {0}, {0}, {0, 1}, {1, 0}};
      else
        data = {{1}, {1}, {1}, {1}, {1}, {1}, {1}, {1}};
    }
    else if (dolfinx::MPI::size(comm) == 3)
    {
      if (rank == 0)
        data = {{2, 0}, {0, 2}, {0}, {0}, {0}, {0}, {0}, {0}, {0}, {0}};
      else if (rank == 1)
        data = {{1}, {1}, {1}, {1}, {1}, {1}, {1}, {1}, {1}, {1, 2}};
      else
        data = {{2, 1}, {2}, {2}, {2}, {2}, {2}, {2}, {2}};
    }
    else
      FAIL("Test only supports <= 3 processes");
    return graph::AdjacencyList<std::int32_t>(std::move(data));
  };

  auto [mesh_fine, parent_cell, parent_facet] = refinement::refine(
      *mesh_coarse, std::nullopt, part_refine, refinement::Option::none);

  auto element = basix::create_element<T>(
      basix::element::family::P, basix::cell::type::interval, 1,
      basix::element::lagrange_variant::unset,
      basix::element::dpc_variant::unset, false);

  auto V_coarse = std::make_shared<fem::FunctionSpace<T>>(
      fem::create_functionspace<T>(mesh_coarse, element, {}));
  auto V_fine
      = std::make_shared<fem::FunctionSpace<T>>(fem::create_functionspace<T>(
          std::make_shared<mesh::Mesh<T>>(mesh_fine), element, {}));

  mesh_fine.topology()->create_connectivity(1, 0);
  mesh_fine.topology()->create_connectivity(0, 1);

  std::vector<int64_t> inclusion_map;
  switch (comm_size)
  {
  case 1:
    inclusion_map = {0, 2, 4, 6, 8};
    break;
  case 2:
    inclusion_map = {0, 2, 4, 6, 8, 10, 12, 14, 16, 18};
    break;
  case 3:
    inclusion_map = {0, 2, 4, 6, 8, 9, 11, 13, 15, 17, 19, 27, 25, 23, 20};
    break;
  }

  CHECK_THAT(transfer::inclusion_mapping(*mesh_coarse, mesh_fine),
             RangeEquals(inclusion_map));

  la::SparsityPattern sp
      = transfer::create_sparsity_pattern(*V_coarse, *V_fine, inclusion_map);

  la::MatrixCSR<T> transfer_matrix(std::move(sp), la::BlockMode::compact);
  transfer::assemble_transfer_matrix<T>(transfer_matrix.mat_set_values(),
                                        *V_coarse, *V_fine, inclusion_map,
                                        weight<T>);

  std::array<std::vector<T>, 3> expected;
  if (comm_size == 1)
  {
    // clang-format off
    expected[0] = {
        /* row_0 */ 1.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        /* row_1 */ 0.0, 0.5, 1.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0,
        /* row_2 */ 0.0, 0.0, 0.0, 0.5, 1.0, 0.5, 0.0, 0.0, 0.0,
        /* row_3 */ 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 1.0, 0.5, 0.0,
        /* row_4 */ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 1.0
    };
    // clang-format on
  }
  else if (comm_size == 2)
  {
    // clang-format off
    expected[0] = {
        /* row_0 */ 1.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        /* row_1 */ 0.0, 0.5, 1.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        /* row_2 */ 0.0, 0.0, 0.0, 0.5, 1.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        /* row_3 */ 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 1.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        /* row_4 */ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 1.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        /* row_5 */ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 1.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        /* row_6 */ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    };
    expected[1] = {
        /* row_0 */ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 1.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0,
        /* row_1 */ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 1.0, 0.5, 0.0, 0.0, 0.0,
        /* row_2 */ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 1.0, 0.5, 0.0,
        /* row_3 */ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 1.0,
        /* row_4 */ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        /* row_5 */ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    };
    // clang-format on
  }
  else if (comm_size == 3)
  {
    // clang-format off
    expected[0] = {
        /* row_0 */ 1.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        /* row_1 */ 0.0, 0.5, 1.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        /* row_2 */ 0.0, 0.0, 0.0, 0.5, 1.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        /* row_3 */ 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 1.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        /* row_4 */ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        /* row_5 */ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        /* row_6 */ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    };
    expected[1] = {
        /* row_0 */ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        /* row_1 */ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 1.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        /* row_2 */ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 1.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        /* row_3 */ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 1.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        /* row_4 */ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 1.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        /* row_5 */ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5,
        /* row_6 */ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    };
    expected[2] = {
        /* row_0 */ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 1.0, 0.5,
        /* row_1 */ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 1.0, 0.5, 0.0, 0.0,
        /* row_2 */ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 1.0, 0.5, 0.0, 0.0, 0.0, 0.0,
        /* row_3 */ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        /* row_4 */ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        /* row_5 */ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        /* row_6 */ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    };
    // clang-format on
  }
  else
  {
    CHECK(false);
  }

  CHECK_THAT(transfer_matrix.to_dense(),
             RangeEquals(expected[rank], [](auto a, auto b)
                         { return std::abs(a - b) <= EPS<T>; }));
}

TEMPLATE_TEST_CASE("Transfer Matrix 2D", "[transfer_matrix]", double)
{
  using T = TestType;

  auto mesh_coarse
      = std::make_shared<mesh::Mesh<T>>(dolfinx::mesh::create_rectangle<T>(
          MPI_COMM_SELF, {{{0, 0}, {1, 1}}}, {1, 1}, mesh::CellType::triangle));
  mesh_coarse->topology()->create_entities(1);

  auto [mesh_fine, parent_cell, parent_facet]
      = refinement::refine(*mesh_coarse, std::nullopt);

  auto element = basix::create_element<T>(
      basix::element::family::P, basix::cell::type::triangle, 1,
      basix::element::lagrange_variant::unset,
      basix::element::dpc_variant::unset, false);

  auto V_coarse = std::make_shared<fem::FunctionSpace<T>>(
      fem::create_functionspace<T>(mesh_coarse, element, {}));
  auto V_fine
      = std::make_shared<fem::FunctionSpace<T>>(fem::create_functionspace<T>(
          std::make_shared<mesh::Mesh<T>>(mesh_fine), element, {}));

  mesh_fine.topology()->create_connectivity(1, 0);
  mesh_fine.topology()->create_connectivity(0, 1);

  std::vector<std::int64_t> inclusion_map
      = transfer::inclusion_mapping(*mesh_coarse, mesh_fine);
  CHECK_THAT(inclusion_map, RangeEquals(std::vector<std::int64_t>{4, 1, 5, 8}));

  la::SparsityPattern sp
      = transfer::create_sparsity_pattern(*V_coarse, *V_fine, inclusion_map);
  la::MatrixCSR<T> transfer_matrix(std::move(sp), la::BlockMode::compact);
  transfer::assemble_transfer_matrix<T>(transfer_matrix.mat_set_values(),
                                        *V_coarse, *V_fine, inclusion_map,
                                        weight<T>);
  transfer_matrix.scatter_rev();
  std::vector<T> expected{0.5, 0.0, 0.5, 0.0, 1.0, 0.0, 0.5, 0.0, 0.0,
                          0.5, 1.0, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0,
                          0.5, 0.0, 0.0, 0.5, 0.0, 1.0, 0.0, 0.5, 0.0,
                          0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.5, 1.0};
  CHECK_THAT(transfer_matrix.to_dense(),
             RangeEquals(expected, [](auto a, auto b)
                         { return std::abs(a - b) <= EPS<T>; }));
}

TEMPLATE_TEST_CASE("Transfer Matrix 3D", "[transfer_matrix]", double)
{
  using T = TestType;

  auto mesh_coarse = std::make_shared<mesh::Mesh<T>>(
      dolfinx::mesh::create_box<T>(MPI_COMM_SELF, {{{0, 0, 0}, {1, 1, 1}}},
                                   {1, 1, 1}, mesh::CellType::tetrahedron));
  mesh_coarse->topology()->create_entities(1);

  auto [mesh_fine, parent_cell, parent_facet]
      = refinement::refine(*mesh_coarse, std::nullopt);

  auto element = basix::create_element<T>(
      basix::element::family::P, basix::cell::type::tetrahedron, 1,
      basix::element::lagrange_variant::unset,
      basix::element::dpc_variant::unset, false);

  auto V_coarse = std::make_shared<fem::FunctionSpace<T>>(
      fem::create_functionspace<T>(mesh_coarse, element, {}));
  auto V_fine
      = std::make_shared<fem::FunctionSpace<T>>(fem::create_functionspace<T>(
          std::make_shared<mesh::Mesh<T>>(mesh_fine), element, {}));

  mesh_fine.topology()->create_connectivity(1, 0);
  mesh_fine.topology()->create_connectivity(0, 1);

  std::vector<std::int64_t> inclusion_map
      = transfer::inclusion_mapping(*mesh_coarse, mesh_fine);
  CHECK_THAT(inclusion_map, RangeEquals(std::vector<std::int64_t>{
                                0, 6, 15, 25, 17, 9, 11, 22}));

  la::SparsityPattern sp
      = transfer::create_sparsity_pattern(*V_coarse, *V_fine, inclusion_map);
  la::MatrixCSR<T> transfer_matrix(std::move(sp), la::BlockMode::compact);
  transfer::assemble_transfer_matrix<T>(transfer_matrix.mat_set_values(),
                                        *V_coarse, *V_fine, inclusion_map,
                                        weight<T>);

  std::vector<T> expected{
      1.0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.0, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
      0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.5,
      0.5, 0.5, 0.0, 1.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.5, 0.0, 0.5, 0.0, 0.0, 0.0,
      0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.5, 0.0, 0.0, 0.0,
      0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.5, 0.5, 1.0, 0.0, 0.0, 0.0, 0.5, 0.0,
      0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
      0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.5, 0.0, 0.5,
      0.5, 1.0, 0.5, 0.0, 0.5, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
      0.5, 0.0, 0.5, 0.0, 0.5, 1.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.5, 0.0, 0.0,
      0.0, 0.5, 0.5, 0.0, 0.0, 0.5, 0.0, 0.0, 0.5, 1.0, 0.0, 0.0, 0.0, 0.5, 0.0,
      0.0, 0.0, 0.0, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0,
      0.0, 0.5, 0.0, 0.0, 0.5, 0.5, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0,
      0.0, 0.0, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0,
      0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.5, 0.5,
      0.5, 1.0, 0.0, 0.0, 0.0, 0.5};
  CHECK_THAT(transfer_matrix.to_dense(),
             RangeEquals(expected, [](auto a, auto b)
                         { return std::abs(a - b) <= EPS<T>; }));
}
