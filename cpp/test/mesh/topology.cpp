// Copyright (C) 2026 Paul T. Kühner
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "mesh/Topology.h"
#include "mesh/cell_types.h"
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_range_equals.hpp>
#include <cstdint>
#include <mpi.h>
#include <vector>

using namespace dolfinx::mesh;

TEST_CASE("Topology duplicates", "[topology][interval]")
{
  MPI_Comm comm = MPI_COMM_WORLD;

  std::vector<CellType> cell_types{CellType::interval};
  std::array<const std::int64_t, 4> cells_interval{0, 1, 0, 1};
  std::vector<std::span<const std::int64_t>> cells{cells_interval};

  std::array<const std::int64_t, 2> original_cell_index_interval{0, 1};
  std::vector<std::span<const std::int64_t>> original_cell_index{
      original_cell_index_interval};
  std::vector<std::span<const int>> ghost_owners{};
  std::vector<std::int64_t> boundary_vertices{};
  create_topology(comm, cell_types, cells, original_cell_index, ghost_owners,
                  boundary_vertices);
}
