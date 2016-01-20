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
// Modified by Benjamin Kehlet 2012
//
// First added:  2007-05-14
// Last changed: 2012-11-12
//
// Unit tests for the mesh library

#include <dolfin.h>
#include <gtest/gtest.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
TEST(SimpleShapesTest, testUnitSquareMesh)
{
  // Create mesh of unit square
  UnitSquareMesh mesh(5, 7);
  ASSERT_EQ(mesh.num_vertices(), (std::size_t) 48);
  ASSERT_EQ(mesh.num_cells(), (std::size_t) 70);
}
//-----------------------------------------------------------------------------
TEST(SimpleShapesTest, testUnitCubeMesh)
{
  // Create mesh of unit cube
  UnitCubeMesh mesh(5, 7, 9);
  ASSERT_EQ(mesh.num_vertices(), (std::size_t) 480);
  ASSERT_EQ(mesh.num_cells(), (std::size_t) 1890);
}
//-----------------------------------------------------------------------------
TEST(MeshRefinement, testRefineUnitSquareMesh)
{
  // Refine mesh of unit square
  UnitSquareMesh mesh0(5, 7);
  Mesh mesh1 = refine(mesh0);
  ASSERT_EQ(mesh1.num_vertices(), (std::size_t) 165);
  ASSERT_EQ(mesh1.num_cells(), (std::size_t) 280);
}
//-----------------------------------------------------------------------------
TEST(MeshRefinement, testRefineUnitCubeMesh)
{
  // Refine mesh of unit cube
  UnitCubeMesh mesh0(5, 7, 9);
  Mesh mesh1 = refine(mesh0);
  ASSERT_EQ(mesh1.num_vertices(), (std::size_t) 3135);
  ASSERT_EQ(mesh1.num_cells(), (std::size_t) 15120);
}
//-----------------------------------------------------------------------------
TEST(MeshIterators, testVertexIterators)
{
  // Iterate over vertices
  UnitCubeMesh mesh(5, 5, 5);
  unsigned int n = 0;
  for (VertexIterator v(mesh); !v.end(); ++v)
    n++;
  ASSERT_EQ(n, mesh.num_vertices());
}
//-----------------------------------------------------------------------------
TEST(MeshIterators, testEdgeIterators)
{
  // Iterate over edges
  UnitCubeMesh mesh(5, 5, 5);
  unsigned int n = 0;
  for (EdgeIterator e(mesh); !e.end(); ++e)
    n++;
  ASSERT_EQ(n, mesh.num_edges());
}
//-----------------------------------------------------------------------------
TEST(MeshIterators, testFaceIterators)
{
  // Iterate over faces
  UnitCubeMesh mesh(5, 5, 5);
  unsigned int n = 0;
  for (FaceIterator f(mesh); !f.end(); ++f)
    n++;
  ASSERT_EQ(n, mesh.num_faces());
}
//-----------------------------------------------------------------------------
TEST(MeshIterators, testFacetIterators)
{
  // Iterate over facets
  UnitCubeMesh mesh(5, 5, 5);
  unsigned int n = 0;
  for (FacetIterator f(mesh); !f.end(); ++f)
    n++;
  ASSERT_EQ(n, mesh.num_facets());
}
//-----------------------------------------------------------------------------
TEST(MeshIterators, testCellIterators)
{
  // Iterate over cells
  UnitCubeMesh mesh(5, 5, 5);
  unsigned int n = 0;
  for (CellIterator c(mesh); !c.end(); ++c)
    n++;
  ASSERT_EQ(n, mesh.num_cells());
}
//-----------------------------------------------------------------------------
TEST(MeshIterators, testMixedIterators)
{
  // Iterate over vertices of cells
  UnitCubeMesh mesh(5, 5, 5);
  unsigned int n = 0;
  for (CellIterator c(mesh); !c.end(); ++c)
    for (VertexIterator v(*c); !v.end(); ++v)
      n++;
  ASSERT_EQ(n, 4*mesh.num_cells());
}
//-----------------------------------------------------------------------------
TEST(BoundaryExtraction, testBoundaryComputation)
{
  // Compute boundary of mesh
  UnitCubeMesh mesh(2, 2, 2);
  BoundaryMesh boundary(mesh, "exterior");
  ASSERT_EQ(boundary.num_vertices(), (std::size_t) 26);
  ASSERT_EQ(boundary.num_cells(), (std::size_t) 48);
}
//-----------------------------------------------------------------------------
TEST(BoundaryExtraction, testBoundaryBoundary)
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
  ASSERT_EQ(b1.num_vertices(), (std::size_t) 0);
  ASSERT_EQ(b1.num_cells(), (std::size_t) 0);
}
//-----------------------------------------------------------------------------
TEST(MeshFunctions, testAssign)
{
  /// Assign value of mesh function
  auto mesh = std::make_shared<UnitSquareMesh>(3, 3);
  MeshFunction<int> f(mesh, 0);
  f[3] = 10;
  Vertex v(*mesh, 3);
  ASSERT_EQ(f[v], 10);
}
//-----------------------------------------------------------------------------
TEST(InputOutput, testMeshXML2D)
{
  // Write and read 2D mesh to/from file
  UnitSquareMesh mesh_out(3, 3);
  Mesh mesh_in;
  File file("unitsquare.xml");
  file << mesh_out;
  file >> mesh_in;
  ASSERT_EQ(mesh_in.num_vertices(), (std::size_t) 16);
}
//-----------------------------------------------------------------------------
TEST(InputOutput, testMeshXML3D)
{
  // Write and read 3D mesh to/from file
  UnitCubeMesh mesh_out(3, 3, 3);
  Mesh mesh_in;
  File file("unitcube.xml");
  file << mesh_out;
  file >> mesh_in;
  ASSERT_EQ(mesh_in.num_vertices(), (std::size_t) 64);
}
//-----------------------------------------------------------------------------
TEST(InputOutput, testMeshFunction)
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
    ASSERT_EQ(f[*v], g[*v]);
}

//-----------------------------------------------
TEST(PyCCInterface, testGetGeometricalDimension)
{
  // Get geometrical dimension of mesh
  UnitSquareMesh mesh(5, 5);
  ASSERT_EQ(mesh.geometry().dim(), (std::size_t) 2);
}
//-----------------------------------------------------------------------------
TEST(PyCCInterface, testGetCoordinates)
{
  // Get coordinates of vertices
  UnitSquareMesh mesh(5, 5);
  ASSERT_EQ(mesh.geometry().num_vertices(), (std::size_t) 36);
}
//-----------------------------------------------------------------------------
TEST(PyCCInterface, testGetCells)
{
  // Get cells of mesh
  UnitSquareMesh mesh(5, 5);
  ASSERT_EQ(mesh.topology().size(2), (std::size_t) 50);
}
//-----------------------------------------------------------------------------
int main(int argc, char **argv)
{
  testing::InitGoogleTest(&argc, argv);

  // FIXME: Only the following test works in Parallel
  // Failed: SimpleShapes; MeshRefinement; BoundaryExtraction
  // MeshFunctions; InputOutput; PyCCInterface
  if (dolfin::MPI::size(MPI_COMM_WORLD) != 1)
    ::testing::GTEST_FLAG(filter) = "MeshIterators.*";

  return RUN_ALL_TESTS();
}
//-----------------------------------------------------------------------------
