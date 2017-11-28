// Copyright (C) 2007 Anders Logg
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
// Unit tests for the mesh library

#include <dolfin.h>
#include <catch.hpp>

using namespace dolfin;

//-----------------------------------------------------------------------------
TEST_CASE("Simple shapes test")
{
  SECTION("Test UnitSquareMesh")
  {
    // Create mesh of unit square
    UnitSquareMesh mesh(5, 7);
    CHECK(mesh.num_vertices() == (std::size_t) 48);
    CHECK(mesh.num_cells() == (std::size_t) 70);

    // Create mesh of unit square
    auto mesh1 = UnitSquareMesh::create({{5, 7}}, CellType::Type::triangle);
    CHECK(mesh1.num_vertices() == (std::size_t) 48);
    CHECK(mesh1.num_cells() == (std::size_t) 70);
  }

  SECTION("Test UnitCubeMesh")
  {
    // Create mesh of unit cube
    UnitCubeMesh mesh(5, 7, 9);
    CHECK(mesh.num_vertices() == (std::size_t) 480);
    CHECK(mesh.num_cells() == (std::size_t) 1890);

    // Create mesh of unit cube
    auto mesh1 = UnitCubeMesh::create({{5, 7, 9}}, CellType::Type::tetrahedron);
    CHECK(mesh1.num_vertices() == (std::size_t) 480);
    CHECK(mesh1.num_cells() == (std::size_t) 1890);
  }
}

TEST_CASE("Mesh refinement")
{
  SECTION("Test refine UnitSquareMesh")
  {
    // Refine mesh of unit square
    UnitSquareMesh mesh0(5, 7);
    Mesh mesh1 = refine(mesh0);
    CHECK(mesh1.num_vertices() == (std::size_t) 165);
    CHECK(mesh1.num_cells() == (std::size_t) 280);
  }

  SECTION("Test refine UnitCubeMesh")
  {
    // Refine mesh of unit cube
    UnitCubeMesh mesh0(5, 7, 9);
    Mesh mesh1 = refine(mesh0);
    CHECK(mesh1.num_vertices() == (std::size_t) 3135);
    CHECK(mesh1.num_cells() == (std::size_t) 15120);
  }
}

TEST_CASE("Mesh iterators")
{
  SECTION("Test vertex iterators")
  {
    // Iterate over vertices
    UnitCubeMesh mesh(5, 5, 5);
    unsigned int n = 0;
    for (VertexIterator v(mesh); !v.end(); ++v)
      n++;

    CHECK(n == mesh.num_vertices());
  }

  SECTION("Test edge iterators")
  {
    // Iterate over edges
    UnitCubeMesh mesh(5, 5, 5);
    unsigned int n = 0;
    for (EdgeIterator e(mesh); !e.end(); ++e)
      n++;

    CHECK(n == mesh.num_edges());
  }

  SECTION("Test face iterators")
  {
    // Iterate over faces
    UnitCubeMesh mesh(5, 5, 5);
    unsigned int n = 0;
    for (FaceIterator f(mesh); !f.end(); ++f)
      n++;

    CHECK(n == mesh.num_faces());
  }

  SECTION("Test facet iterators")
  {
    // Iterate over facets
    UnitCubeMesh mesh(5, 5, 5);
    unsigned int n = 0;
    for (FacetIterator f(mesh); !f.end(); ++f)
      n++;

    CHECK(n == mesh.num_facets());
  }

  SECTION("Test cell iterators")
  {
    // Iterate over cells
    UnitCubeMesh mesh(5, 5, 5);
    unsigned int n = 0;
    for (CellIterator c(mesh); !c.end(); ++c)
      n++;

    CHECK(n == mesh.num_cells());
  }

  SECTION("Test mixed iterators")
  {
    // Iterate over vertices of cells
    UnitCubeMesh mesh(5, 5, 5);
    unsigned int n = 0;
    for (CellIterator c(mesh); !c.end(); ++c)
      for (VertexIterator v(*c); !v.end(); ++v)
        n++;

    CHECK(n == 4*mesh.num_cells());
  }

  SECTION("Test boundary computation")
  {
    // Compute boundary of mesh
    UnitCubeMesh mesh(2, 2, 2);
    BoundaryMesh boundary(mesh, "exterior");
    CHECK(boundary.num_vertices() == (std::size_t) 26);
    CHECK(boundary.num_cells() == (std::size_t) 48);
  }
}

TEST_CASE("Boundary extraction")
{
  SECTION("Test boundary of boundary")
  {
    // Compute boundary of boundary
    //
    // Note that we can't do
    //
    //   BoundaryMesh b0(mesh);
    //   BoundaryMesh b1(b0);
    //
    // since b1 would then be a copy of b0 (copy
    // constructor in Mesh will be used).

    UnitCubeMesh mesh(2, 2, 2);
    BoundaryMesh b0(mesh, "exterior");
    b0.order();
    BoundaryMesh b1(b0, "exterior");
    CHECK(b1.num_vertices() == (std::size_t) 0);
    CHECK(b1.num_cells() == (std::size_t) 0);
  }

  SECTION("Test assign")
  {
    /// Assign value of mesh function
    auto mesh = std::make_shared<UnitSquareMesh>(3, 3);
    MeshFunction<int> f(mesh, 0);
    f[3] = 10;
    Vertex v(*mesh, 3);
    CHECK(f[v] == 10);
  }
}

TEST_CASE("InputOutput")
{
  SECTION("Test mesh XML 2D")
  {
    // Write and read 2D mesh to/from file
    UnitSquareMesh mesh_out(3, 3);
    Mesh mesh_in;
    File file("unitsquare.xml");
    file << mesh_out;
    file >> mesh_in;
    CHECK(mesh_in.num_vertices() == (std::size_t) 16);
  }

  SECTION("Test mesh XML 3D")
  {
    // Write and read 3D mesh to/from file
    UnitCubeMesh mesh_out(3, 3, 3);
    Mesh mesh_in;
    File file("unitcube.xml");
    file << mesh_out;
    file >> mesh_in;
    CHECK(mesh_in.num_vertices() == (std::size_t) 64);
  }

  SECTION("Test MeshFunction")
  {
    // Write and read mesh function to/from file
    auto mesh = std::make_shared<UnitSquareMesh>(1, 1);
    MeshFunction<int> f(mesh, 0);
    f[0] = 2;
    f[1] = 4;
    f[2] = 6;
    f[3] = 8;
    File file("meshfunction.xml");
    file << f;
    MeshFunction<int> g(mesh, 0);
    file >> g;
    for (VertexIterator v(*mesh); !v.end(); ++v)
      CHECK(f[*v] == g[*v]);
  }
}

TEST_CASE("PyCCInterface")
{
  SECTION("Test get geometrical dimension")
  {
    // Get geometrical dimension of mesh
    UnitSquareMesh mesh(5, 5);
    CHECK(mesh.geometry().dim() == (std::size_t) 2);
  }

  SECTION("Test get coordinates")
  {
    // Get coordinates of vertices
    UnitSquareMesh mesh(5, 5);
    CHECK(mesh.geometry().num_vertices() == (std::size_t) 36);
  }

  SECTION("Test get cells")
  {
    // Get cells of mesh
    UnitSquareMesh mesh(5, 5);
    CHECK(mesh.topology().size(2) == (std::size_t) 50);
  }
}
