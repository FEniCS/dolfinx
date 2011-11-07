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
// First added:  2007-05-14
// Last changed: 2011-10-30
//
// Unit tests for the mesh library

#include <dolfin.h>
#include <dolfin/common/unittest.h>

using namespace dolfin;

class SimpleShapes : public CppUnit::TestFixture
{
  CPPUNIT_TEST_SUITE(SimpleShapes);
  CPPUNIT_TEST(testUnitSquare);
  CPPUNIT_TEST(testUnitCube);
  CPPUNIT_TEST_SUITE_END();

public:

  void testUnitSquare()
  {
    // Create mesh of unit square
    UnitSquare mesh(5, 7);
    CPPUNIT_ASSERT(mesh.num_vertices() == 48);
    CPPUNIT_ASSERT(mesh.num_cells() == 70);
  }

  void testUnitCube()
  {
    // Create mesh of unit cube
    UnitCube mesh(5, 7, 9);
    CPPUNIT_ASSERT(mesh.num_vertices() == 480);
    CPPUNIT_ASSERT(mesh.num_cells() == 1890);
  }

};

class MeshRefinement : public CppUnit::TestFixture
{
  CPPUNIT_TEST_SUITE(MeshRefinement);
  CPPUNIT_TEST(testRefineUnitSquare);
  CPPUNIT_TEST(testRefineUnitCube);
  CPPUNIT_TEST_SUITE_END();

public:

  void testRefineUnitSquare()
  {
    // Refine mesh of unit square
    UnitSquare mesh0(5, 7);
    Mesh mesh1 = refine(mesh0);
    CPPUNIT_ASSERT(mesh1.num_vertices() == 165);
    CPPUNIT_ASSERT(mesh1.num_cells() == 280);
  }

  void testRefineUnitCube()
  {
    // Refine mesh of unit cube
    UnitCube mesh0(5, 7, 9);
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
    UnitCube mesh(5, 5, 5);
    unsigned int n = 0;
    for (VertexIterator v(mesh); !v.end(); ++v)
      n++;
    CPPUNIT_ASSERT(n == mesh.num_vertices());
  }

  void testEdgeIterators()
  {
    // Iterate over edges
    UnitCube mesh(5, 5, 5);
    unsigned int n = 0;
    for (EdgeIterator e(mesh); !e.end(); ++e)
      n++;
    CPPUNIT_ASSERT(n == mesh.num_edges());
  }

  void testFaceIterators()
  {
    // Iterate over faces
    UnitCube mesh(5, 5, 5);
    unsigned int n = 0;
    for (FaceIterator f(mesh); !f.end(); ++f)
      n++;
    CPPUNIT_ASSERT(n == mesh.num_faces());
  }

  void testFacetIterators()
  {
    // Iterate over facets
    UnitCube mesh(5, 5, 5);
    unsigned int n = 0;
    for (FacetIterator f(mesh); !f.end(); ++f)
      n++;
    CPPUNIT_ASSERT(n == mesh.num_facets());
  }

  void testCellIterators()
  {
    // Iterate over cells
    UnitCube mesh(5, 5, 5);
    unsigned int n = 0;
    for (CellIterator c(mesh); !c.end(); ++c)
      n++;
    CPPUNIT_ASSERT(n == mesh.num_cells());
  }

  void testMixedIterators()
  {
    // Iterate over vertices of cells
    UnitCube mesh(5, 5, 5);
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
    UnitCube mesh(2, 2, 2);
    BoundaryMesh boundary(mesh);
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

    UnitCube mesh(2, 2, 2);
    BoundaryMesh b0(mesh);
    BoundaryMesh b1;
    b0.order();
    b1.init_exterior_boundary(b0);
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
    UnitSquare mesh(3, 3);
    MeshFunction<int> f(mesh, 0);
    f[3] = 10;
    Vertex v(mesh, 3);
    CPPUNIT_ASSERT(f[v] == 10);
  }

};

class MeshValueCollections : public CppUnit::TestFixture
{
  CPPUNIT_TEST_SUITE(MeshValueCollections);
  CPPUNIT_TEST(testAssign2DCells);
  CPPUNIT_TEST(testAssign2DFacets);
  CPPUNIT_TEST(testAssign2DVertices);
  CPPUNIT_TEST(testMeshFunctionAssign2DCells);
  CPPUNIT_TEST(testMeshFunctionAssign2DFacets);
  CPPUNIT_TEST(testMeshFunctionAssign2DVertices);
  CPPUNIT_TEST_SUITE_END();

public:

  void testAssign2DCells()
  {
    UnitSquare mesh(3, 3);
    const dolfin::uint ncells = mesh.num_cells();
    MeshValueCollection<int> f(2);
    bool all_new = true;
    for (CellIterator cell(mesh); !cell.end(); ++cell)
    {
      bool this_new;
      const int value = ncells - cell->index();
      this_new = f.set_value(cell->index(), value, mesh);
      all_new = all_new && this_new;
    }
    MeshValueCollection<int> g(2);
    g = f;
    CPPUNIT_ASSERT_EQUAL(ncells, f.size());
    CPPUNIT_ASSERT_EQUAL(ncells, g.size());
    CPPUNIT_ASSERT(all_new);
    for (CellIterator cell(mesh); !cell.end(); ++cell)
    {
      const int value = ncells - cell->index();
      CPPUNIT_ASSERT_EQUAL(value, g.get_value(cell->index(), 0));
    }
  }

  void testAssign2DFacets()
  {
    UnitSquare mesh(3, 3);
    mesh.init(2,1);
    const dolfin::uint ncells = mesh.num_cells();
    MeshValueCollection<int> f(1);
    bool all_new = true;
    for (CellIterator cell(mesh); !cell.end(); ++cell)
    {
      const int value = ncells - cell->index();
      for (dolfin::uint i = 0; i < cell->num_entities(1); ++i)
      {
        bool this_new;
        this_new = f.set_value(cell->index(), i, value+i);
        all_new = all_new && this_new;
      }
    }
    MeshValueCollection<int> g(1);
    g = f;
    CPPUNIT_ASSERT_EQUAL(ncells*3, f.size());
    CPPUNIT_ASSERT_EQUAL(ncells*3, g.size());
    CPPUNIT_ASSERT(all_new);
    for (CellIterator cell(mesh); !cell.end(); ++cell)
    {
      for (dolfin::uint i = 0; i < cell->num_entities(1); ++i)
      {
        const int value = ncells - cell->index() + i;
        CPPUNIT_ASSERT_EQUAL(value, g.get_value(cell->index(), i));
      }
    }
  }

  void testAssign2DVertices()
  {
    UnitSquare mesh(3, 3);
    mesh.init(2,0);
    const dolfin::uint ncells = mesh.num_cells();
    MeshValueCollection<int> f(0);
    bool all_new = true;
    for (CellIterator cell(mesh); !cell.end(); ++cell)
    {
      const int value = ncells - cell->index();
      for (dolfin::uint i = 0; i < cell->num_entities(0); ++i)
      {
        bool this_new;
        this_new = f.set_value(cell->index(), i, value+i);
        all_new = all_new && this_new;
      }
    }
    MeshValueCollection<int> g(0);
    g = f;
    CPPUNIT_ASSERT_EQUAL(ncells*3, f.size());
    CPPUNIT_ASSERT_EQUAL(ncells*3, g.size());
    CPPUNIT_ASSERT(all_new);
    for (CellIterator cell(mesh); !cell.end(); ++cell)
    {
      for (dolfin::uint i = 0; i < cell->num_entities(0); ++i)
      {
        const int value = ncells - cell->index() + i;
        CPPUNIT_ASSERT_EQUAL(value, g.get_value(cell->index(), i));
      }
    }
  }

  void testMeshFunctionAssign2DCells()
  {
    UnitSquare mesh(3, 3);
    const dolfin::uint ncells = mesh.num_cells();
    MeshFunction<int> f(mesh, 2, 0);
    for (CellIterator cell(mesh); !cell.end(); ++cell)
    {
      f[cell->index()] = ncells - cell->index();
    }
    MeshValueCollection<int> g(2);
    g = f;
    CPPUNIT_ASSERT_EQUAL(ncells, f.size());
    CPPUNIT_ASSERT_EQUAL(ncells, g.size());
    for (CellIterator cell(mesh); !cell.end(); ++cell)
    {
      const int value = ncells - cell->index();
      CPPUNIT_ASSERT_EQUAL(value, g.get_value(cell->index(), 0));
    }
  }

  void testMeshFunctionAssign2DFacets()
  {
    UnitSquare mesh(3, 3);
    mesh.init(1);
    MeshFunction<int> f(mesh, 1, 25);
    MeshValueCollection<int> g(1);
    g = f;
    CPPUNIT_ASSERT_EQUAL(mesh.num_facets(), f.size());
    CPPUNIT_ASSERT_EQUAL(mesh.num_cells()*3, g.size());
    for (CellIterator cell(mesh); !cell.end(); ++cell)
    {
      for (dolfin::uint i = 0; i < cell->num_entities(1); ++i)
      {
        CPPUNIT_ASSERT_EQUAL(25, g.get_value(cell->index(), i));
      }
    }
  }

  void testMeshFunctionAssign2DVertices()
  {
    UnitSquare mesh(3, 3);
    mesh.init(0);
    MeshFunction<int> f(mesh, 0, 25);
    MeshValueCollection<int> g(0);
    g = f;
    CPPUNIT_ASSERT_EQUAL(mesh.num_vertices(), f.size());
    CPPUNIT_ASSERT_EQUAL(mesh.num_cells()*3, g.size());
    for (CellIterator cell(mesh); !cell.end(); ++cell)
    {
      for (dolfin::uint i = 0; i < cell->num_entities(0); ++i)
      {
        CPPUNIT_ASSERT_EQUAL(25, g.get_value(cell->index(), i));
      }
    }
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
    UnitSquare mesh_out(3, 3);
    Mesh mesh_in;
    File file("unitsquare.xml");
    file << mesh_out;
    file >> mesh_in;
    CPPUNIT_ASSERT(mesh_in.num_vertices() == 16);
  }

  void testMeshXML3D()
  {
    // Write and read 3D mesh to/from file
    UnitCube mesh_out(3, 3, 3);
    Mesh mesh_in;
    File file("unitcube.xml");
    file << mesh_out;
    file >> mesh_in;
    CPPUNIT_ASSERT(mesh_in.num_vertices() == 64);
  }

  void testMeshFunction()
  {
    // Write and read mesh function to/from file
    UnitSquare mesh(1, 1);
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
    UnitSquare mesh(5, 5);
    CPPUNIT_ASSERT(mesh.geometry().dim() == 2);
  }

  void testGetCoordinates()
  {
    // Get coordinates of vertices
    UnitSquare mesh(5, 5);
    CPPUNIT_ASSERT(mesh.geometry().size() == 36);
  }

  void testGetCells()
  {
    // Get cells of mesh
    UnitSquare mesh(5, 5);
    CPPUNIT_ASSERT(mesh.topology().size(2) == 50);
  }

};

int main()
{
  CPPUNIT_TEST_SUITE_REGISTRATION(MeshIterators);
  CPPUNIT_TEST_SUITE_REGISTRATION(MeshValueCollections);

  // FIXME: The following test breaks in parallel
  if (dolfin::MPI::num_processes() == 1)
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
