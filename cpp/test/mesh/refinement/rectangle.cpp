// Copyright (C) 2024 Paul T. Kühner
//
// This file is part of DOLFINX (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include <cstddef>
#include <limits>
#include <optional>

#include <mpi.h>

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <catch2/matchers/catch_matchers_range_equals.hpp>

#include <dolfinx/common/MPI.h>
#include <dolfinx/graph/AdjacencyList.h>
#include <dolfinx/mesh/Mesh.h>
#include <dolfinx/mesh/generation.h>
#include <dolfinx/mesh/utils.h>
#include <dolfinx/refinement/interval.h>
#include <dolfinx/refinement/refine.h>

using namespace dolfinx;
using namespace Catch::Matchers;

namespace
{
template <typename T>
void CHECK_adjacency_list_equal(
    const dolfinx::graph::AdjacencyList<T>& adj_list,
    const std::vector<std::vector<T>>& expected_list)
{
  REQUIRE(static_cast<std::size_t>(adj_list.num_nodes())
          == expected_list.size());

  for (T i = 0; i < adj_list.num_nodes(); i++)
  {
    CHECK_THAT(adj_list.links(i),
               Catch::Matchers::RangeEquals(expected_list[i]));
  }
}
template <typename T>
constexpr auto EPS = std::numeric_limits<T>::epsilon();
} // namespace

TEMPLATE_TEST_CASE("Rectangle uniform refinement",
                   "refinement,rectangle,uniform", double) // TODO: fix float
{
  /**
To Debug this test or visualize use:

from mpi4py import MPI

import febug
import pyvista

import dolfinx

mesh = dolfinx.mesh.create_rectangle(
    MPI.COMM_WORLD,
    ((0.0, 0.0), (1.0, 1.0)),
    (1, 1),
    dolfinx.mesh.CellType.triangle,
    ghost_mode=dolfinx.mesh.GhostMode.shared_facet,
)

mesh.topology.create_entities(1)

(
    refined,
    _,
    _,
) = dolfinx.mesh.refine(mesh, redistribute=False)

plotter = pyvista.Plotter()

refined.topology.create_connectivity(0, 2)

for tdim in range(refined.topology.dim):
    refined.topology.create_entities(tdim)
    febug.plot_entity_indices(refined, tdim, plotter=plotter)

plotter.show()

  */
  using T = TestType;

  if (dolfinx::MPI::size(MPI_COMM_WORLD) > 1)
    return;

  // Mesh connectivity tested/available in mesh/rectangle.cpp
  mesh::Mesh<T> mesh = dolfinx::mesh::create_rectangle<T>(
      MPI_COMM_SELF, {{{0, 0}, {1, 1}}}, {1, 1}, mesh::CellType::triangle);

  // plaza requires the edges to be pre initialized!
  mesh.topology()->create_entities(1);

  auto [mesh_fine, parent_cell, parent_facet]
      = refinement::refine(mesh, std::nullopt, false, mesh::GhostMode::none,
                           refinement::Option::parent_cell_and_facet);

  // vertex layout:
  // 8---7---5
  // |\  |  /|
  // | \ | / |
  // |  \|/  |
  // 6---0---3
  // |  /|\  |
  // | / | \ |
  // |/  |  \|
  // 4---2---1

  std::vector<T> expected_x = {/* v_0 */ 0.5, 0.5, 0.0,
                               /* v_1 */ 1.0, 0.0, 0.0,
                               /* v_2 */ 0.5, 0.0, 0.0,
                               /* v_3 */ 1.0, 0.5, 0.0,
                               /* v_4 */ 0.0, 0.0, 0.0,
                               /* v_5 */ 1.0, 1.0, 0.0,
                               /* v_6 */ 0.0, 0.5, 0.0,
                               /* v_7 */ 0.5, 1.0, 0.0,
                               /* v_8 */ 0.0, 1.0, 0.0};

  CHECK_THAT(mesh_fine.geometry().x(),
             RangeEquals(expected_x, [](auto a, auto b)
                         { return std::abs(a - b) <= EPS<T>; }));

  // edge layout:
  // x---x---x
  // |\  |  /|
  // | \ | / |
  // |  \|/  |
  // x---x---x
  // |  /|\  |
  // | / | \ |
  // |/  |  \|
  // x---x---x
  mesh_fine.topology()->create_connectivity(1, 0);
  auto e_to_v = mesh_fine.topology()->connectivity(1, 0);
  REQUIRE(e_to_v);

  CHECK_adjacency_list_equal(*e_to_v, {/* e_0 */ {0, 1},
                                       /* e_1 */ {0, 2},
                                       /* e_2 */ {0, 3},
                                       /* e_3 */ {0, 4},
                                       /* e_4 */ {0, 5},
                                       /* e_5 */ {0, 6},
                                       /* e_6 */ {0, 7},
                                       /* e_7 */ {0, 8},
                                       /* e_8 */ {1, 2},
                                       /* e_9 */ {1, 3},
                                       /* e_10 */ {2, 4},
                                       /* e_11 */ {3, 5},
                                       /* e_12 */ {4, 6},
                                       /* e_13 */ {5, 7},
                                       /* e_14 */ {6, 8},
                                       /* e_15 */ {7, 8}});
}