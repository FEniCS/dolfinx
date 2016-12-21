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
#include <gtest/gtest.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
TEST(MeshColoring, test_mesh_coloring)
{
  // Create mesh
  auto mesh = std::make_shared<UnitCubeMesh>(24, 24, 24);

  // Compute vertex-based coloring
  mesh->color("vertex");
  const MeshFunction<std::size_t> colors_vertex
    = MeshColoring::cell_colors(mesh, "vertex");
  plot(colors_vertex, "Vertex-based cell coloring");

  // Compute edge-based coloring
  mesh->color("edge");
  const CellFunction<std::size_t> colors_edge
    = MeshColoring::cell_colors(mesh, "edge");
  plot(colors_edge, "Edge-based cell coloring");

  // Compute facet-based coloring
  mesh->color("facet");
  const CellFunction<std::size_t> colors_facet
    = MeshColoring::cell_colors(mesh, "facet");
  plot(colors_facet, "Facet-based cell coloring");

  // Compute facet-based coloring with distance 2
  std::vector<std::size_t> coloring_type
    = {{mesh->topology().dim(),
        mesh->topology().dim() - 1,
        mesh->topology().dim(),
        mesh->topology().dim() - 1,
        mesh->topology().dim()}};
  mesh->color(coloring_type);
  const CellFunction<std::size_t> colors_vertex_2
    = MeshColoring::cell_colors(mesh, coloring_type);
  plot(colors_vertex_2, "Facet-based cell coloring with distance 2");
}

// Test all
int MeshColoring_main(int argc, char **argv)
{
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
