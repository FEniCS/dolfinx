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
#include <dolfin/common/unittest.h>

using namespace dolfin;

class SimpleShapes : public CppUnit::TestFixture
{
  CPPUNIT_TEST_SUITE(SimpleShapes);
  CPPUNIT_TEST(testUnitSquareMesh);
  CPPUNIT_TEST(testUnitCubeMesh);
  CPPUNIT_TEST_SUITE_END();

public:

  void testUnitSquareMesh()
  {
    // Create mesh of unit square
    UnitSquareMesh mesh(5, 7);
    CPPUNIT_ASSERT(mesh.num_vertices() == 48);
    CPPUNIT_ASSERT(mesh.num_cells() == 70);
  }

  void testUnitCubeMesh()
  {
    // Create mesh of unit cube
    UnitCubeMesh mesh(5, 7, 9);
    CPPUNIT_ASSERT(mesh.num_vertices() == 480);
    CPPUNIT_ASSERT(mesh.num_cells() == 1890);
  }

};

class MeshRefinement : public CppUnit::TestFixture
{
  CPPUNIT_TEST_SUITE(MeshRefinement);
  CPPUNIT_TEST(testRefineUnitSquareMesh);
  CPPUNIT_TEST(testRefineUnitCubeMesh);
  CPPUNIT_TEST_SUITE_END();

public:

  void testRefineUnitSquareMesh()
  {
    // Refine mesh of unit square
    UnitSquareMesh mesh0(5, 7);
    Mesh mesh1 = refine(mesh0);
    CPPUNIT_ASSERT(mesh1.num_vertices() == 165);
    CPPUNIT_ASSERT(mesh1.num_cells() == 280);
  }

  void testRefineUnitCubeMesh()
  {
    // Refine mesh of unit cube
    UnitCubeMesh mesh0(5, 7, 9);
    Mesh mesh1 = refine(mesh0);
    CPPUNIT_ASSERT(mesh1.num_vertices() == 3135);
    CPPUNIT_ASSERT(mesh1.num_cells() == 15120);
  }

};

class MeshIterators : public CppUnit::TestFixture
{
  CPPUNIT_TEST_SUITE(MeshIterators);
  CPPUNIT_TEST(testVertexIterators);
  CPPUNIT_TEST(testEdgeIterators);
  CPPUNIT_TEST(testFaceIterators);
  CPPUNIT_TEST(testFacetIterators);
  CPPUNIT_TEST(testCellIterators);
  CPPUNIT_TEST(testMixedIterators);
  CPPUNIT_TEST_SUITE_END();

public:

  void testVertexIterators()
  {
    // Iterate over vertices
    UnitCubeMesh mesh(5, 5, 5);
    unsigned int n = 0;
    for (VertexIterator v(mesh); !v.end(); ++v)
      n++;
    CPPUNIT_ASSERT(n == mesh.num_vertices());
  }

  void testEdgeIterators()
  {
    // Iterate over edges
    UnitCubeMesh mesh(5, 5, 5);
    unsigned int n = 0;
    for (EdgeIterator e(mesh); !e.end(); ++e)
      n++;
    CPPUNIT_ASSERT(n == mesh.num_edges());
  }

  void testFaceIterators()
  {
    // Iterate over faces
    UnitCubeMesh mesh(5, 5, 5);
    unsigned int n = 0;
    for (FaceIterator f(mesh); !f.end(); ++f)
      n++;
    CPPUNIT_ASSERT(n == mesh.num_faces());
  }

  void testFacetIterators()
  {
    // Iterate over facets
    UnitCubeMesh mesh(5, 5, 5);
    unsigned int n = 0;
    for (FacetIterator f(mesh); !f.end(); ++f)
      n++;
    CPPUNIT_ASSERT(n == mesh.num_facets());
  }

  void testCellIterators()
  {
    // Iterate over cells
    UnitCubeMesh mesh(5, 5, 5);
    unsigned int n = 0;
    for (CellIterator c(mesh); !c.end(); ++c)
      n++;
    CPPUNIT_ASSERT(n == mesh.num_cells());
  }

  void testMixedIterators()
  {
    // Iterate over vertices of cells
    UnitCubeMesh mesh(5, 5, 5);
    unsigned int n = 0;
    for (CellIterator c(mesh); !c.end(); ++c)
      for (VertexIterator v(*c); !v.end(); ++v)
        n++;
    CPPUNIT_ASSERT(n == 4*mesh.num_cells());
  }

};

class BoundaryExtraction : public CppUnit::TestFixture
{
  CPPUNIT_TEST_SUITE(BoundaryExtraction);
  CPPUNIT_TEST(testBoundaryComputation);
  CPPUNIT_TEST(testBoundaryBoundary);
  CPPUNIT_TEST_SUITE_END();

public:

  void testBoundaryComputation()
  {
    // Compute boundary of mesh
    UnitCubeMesh mesh(2, 2, 2);
    BoundaryMesh boundary(mesh, "exterior");
    CPPUNIT_ASSERT(boundary.num_vertices() == 26);
    CPPUNIT_ASSERT(boundary.num_cells() == 48);
  }

  void testBoundaryBoundary()
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
    CPPUNIT_ASSERT(b1.num_vertices() == 0);
    CPPUNIT_ASSERT(b1.num_cells() == 0);
  }

};

class MeshFunctions : public CppUnit::TestFixture
{
  CPPUNIT_TEST_SUITE(MeshFunctions);
  CPPUNIT_TEST(testAssign);
  CPPUNIT_TEST_SUITE_END();

public:

  void testAssign()
  {
    /// Assign value of mesh function
    UnitSquareMesh mesh(3, 3);
    MeshFunction<int> f(mesh, 0);
    f[3] = 10;
    Vertex v(mesh, 3);
    CPPUNIT_ASSERT(f[v] == 10);
  }

};

class InputOutput : public CppUnit::TestFixture
{
  CPPUNIT_TEST_SUITE(InputOutput);
  CPPUNIT_TEST(testMeshXML2D);
  CPPUNIT_TEST(testMeshXML3D);
  CPPUNIT_TEST(testMeshFunction);
  CPPUNIT_TEST_SUITE_END();

public:

  void testMeshXML2D()
  {
    // Write and read 2D mesh to/from file
    UnitSquareMesh mesh_out(3, 3);
    Mesh mesh_in;
    File file("unitsquare.xml");
    file << mesh_out;
    file >> mesh_in;
    CPPUNIT_ASSERT(mesh_in.num_vertices() == 16);
  }

  void testMeshXML3D()
  {
    // Write and read 3D mesh to/from file
    UnitCubeMesh mesh_out(3, 3, 3);
    Mesh mesh_in;
    File file("unitcube.xml");
    file << mesh_out;
    file >> mesh_in;
    CPPUNIT_ASSERT(mesh_in.num_vertices() == 64);
  }

  void testMeshFunction()
  {
    // Write and read mesh function to/from file
    UnitSquareMesh mesh(1, 1);
    MeshFunction<int> f(mesh, 0);
    f[0] = 2;
    f[1] = 4;
    f[2] = 6;
    f[3] = 8;
    File file("meshfunction.xml");
    file << f;
    MeshFunction<int> g(mesh, 0);
    file >> g;
    for (VertexIterator v(mesh); !v.end(); ++v)
      CPPUNIT_ASSERT(f[*v] == g[*v]);
  }

};

class PyCCInterface : public CppUnit::TestFixture
{
  CPPUNIT_TEST_SUITE(PyCCInterface);
  CPPUNIT_TEST(testGetGeometricalDimension);
  CPPUNIT_TEST(testGetCoordinates);
  CPPUNIT_TEST(testGetCells);
  CPPUNIT_TEST_SUITE_END();

public:

  void testGetGeometricalDimension()
  {
    // Get geometrical dimension of mesh
    UnitSquareMesh mesh(5, 5);
    CPPUNIT_ASSERT(mesh.geometry().dim() == 2);
  }

  void testGetCoordinates()
  {
    // Get coordinates of vertices
    UnitSquareMesh mesh(5, 5);
    CPPUNIT_ASSERT(mesh.geometry().num_vertices() == 36);
  }

  void testGetCells()
  {
    // Get cells of mesh
    UnitSquareMesh mesh(5, 5);
    CPPUNIT_ASSERT(mesh.topology().size(2) == 50);
  }

};

int main()
{
  CPPUNIT_TEST_SUITE_REGISTRATION(MeshIterators);

  // FIXME: The following test breaks in parallel
  if (dolfin::MPI::size(MPI_COMM_WORLD) == 1)
  {
    CPPUNIT_TEST_SUITE_REGISTRATION(SimpleShapes);
    CPPUNIT_TEST_SUITE_REGISTRATION(MeshRefinement);
    CPPUNIT_TEST_SUITE_REGISTRATION(BoundaryExtraction);
    CPPUNIT_TEST_SUITE_REGISTRATION(MeshFunctions);
    CPPUNIT_TEST_SUITE_REGISTRATION(InputOutput);
    CPPUNIT_TEST_SUITE_REGISTRATION(PyCCInterface);
  }

  DOLFIN_TEST;
}
