// Copyright (C) 2016 Garth N. Wells
//
// This file is part of DOLFIN.
//
// DOLFIN is free software: you can redistribute it and/or modify
// it under the terms of the GNU Lesser General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// DOLFIN is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with DOLFIN. If not, see <http://www.gnu.org/licenses/>.
//
// Unit tests for MeshColoring

#include <dolfin.h>
#include <catch.hpp>

using namespace dolfin;

TEST_CASE("MeshColoring")
{

  SECTION("mesh coloring computation")
  {
    // Create mesh
    auto mesh = std::make_shared<UnitCubeMesh>(24, 24, 24);

    // Compute vertex-based coloring
    mesh->color("vertex");
    const MeshFunction<std::size_t> colors_vertex
      = MeshColoring::cell_colors(mesh, "vertex");

    // Compute edge-based coloring
    mesh->color("edge");
    const MeshFunction<std::size_t> colors_edge
      = MeshColoring::cell_colors(mesh, "edge");

    // Compute facet-based coloring
    mesh->color("facet");
    const MeshFunction<std::size_t> colors_facet
      = MeshColoring::cell_colors(mesh, "facet");

    // Compute facet-based coloring with distance 2
    std::vector<std::size_t> coloring_type
      = {{mesh->topology().dim(),
          mesh->topology().dim() - 1,
          mesh->topology().dim(),
          mesh->topology().dim() - 1,
          mesh->topology().dim()}};
    mesh->color(coloring_type);
    const MeshFunction<std::size_t> colors_vertex_2
      = MeshColoring::cell_colors(mesh, coloring_type);
  }
}
