// Copyright (C) 2007 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2007-05-14
// Last changed: 2007-05-24
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
    CPPUNIT_ASSERT(mesh.numVertices() == 48);
    CPPUNIT_ASSERT(mesh.numCells() == 70);
  }
   
  void testUnitCube()
  {
    // Create mesh of unit cube
    UnitCube mesh(5, 7, 9);
    CPPUNIT_ASSERT(mesh.numVertices() == 480);
    CPPUNIT_ASSERT(mesh.numCells() == 1890);
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
    UnitSquare mesh(5, 7);
    mesh.refine();
    CPPUNIT_ASSERT(mesh.numVertices() == 165);
    CPPUNIT_ASSERT(mesh.numCells() == 280);
  }
  
  void testRefineUnitCube()
  {
    // Refine mesh of unit cube
    UnitCube mesh(5, 7, 9);
    mesh.refine();
    CPPUNIT_ASSERT(mesh.numVertices() == 3135);
    CPPUNIT_ASSERT(mesh.numCells() == 15120);
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
    CPPUNIT_ASSERT(n == mesh.numVertices());
  }

  void testEdgeIterators()
  {
    // Iterate over edges
    UnitCube mesh(5, 5, 5);
    unsigned int n = 0;
    for (EdgeIterator e(mesh); !e.end(); ++e)
      n++;
    CPPUNIT_ASSERT(n == mesh.numEdges());
  }

  void testFaceIterators()
  {
    // Iterate over faces
    UnitCube mesh(5, 5, 5);
    unsigned int n = 0;
    for (FaceIterator f(mesh); !f.end(); ++f)
      n++;
    CPPUNIT_ASSERT(n == mesh.numFaces());
  }

  void testFacetIterators()
  {
    // Iterate over facets
    UnitCube mesh(5, 5, 5);
    unsigned int n = 0;
    for (FacetIterator f(mesh); !f.end(); ++f)
      n++;
    CPPUNIT_ASSERT(n == mesh.numFacets());
  }

  void testCellIterators()
  {
    // Iterate over cells
    UnitCube mesh(5, 5, 5);
    unsigned int n = 0;
    for (CellIterator c(mesh); !c.end(); ++c)
      n++;
    CPPUNIT_ASSERT(n == mesh.numCells());
  }
        
  void testMixedIterators()
  {
    // Iterate over vertices of cells
    UnitCube mesh(5, 5, 5);
    unsigned int n = 0;
    for (CellIterator c(mesh); !c.end(); ++c)
      for (VertexIterator v(*c); !v.end(); ++v)
        n++;
    CPPUNIT_ASSERT(n == 4*mesh.numCells());
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
    CPPUNIT_ASSERT(boundary.numVertices() == 26);
    CPPUNIT_ASSERT(boundary.numCells() == 48);
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
    b1.init(b0);
    CPPUNIT_ASSERT(b1.numVertices() == 0);
    CPPUNIT_ASSERT(b1.numCells() == 0);
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
    f.set(3, 10);
    Vertex v(mesh, 3);
    CPPUNIT_ASSERT(f(v) == 10);
  }
  
};
      
class InputOutput : public CppUnit::TestFixture
{
  CPPUNIT_TEST_SUITE(InputOutput);
  CPPUNIT_TEST(testMeshXML2D);
  CPPUNIT_TEST(testMeshXML3D);
  CPPUNIT_TEST(testMeshMatlab2D);
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
    CPPUNIT_ASSERT(mesh_in.numVertices() == 16);
  }
  
  void testMeshXML3D()
  {
    // Write and read 3D mesh to/from file
    UnitCube mesh_out(3, 3, 3);
    Mesh mesh_in;
    File file("unitcube.xml");
    file << mesh_out;
    file >> mesh_in;
    CPPUNIT_ASSERT(mesh_in.numVertices() == 64);
  }

  void testMeshMatlab2D()
  {
    // Write matlab format (no real test)
    UnitSquare mesh(5, 5);
    File file("unitsquare.m");
    file << mesh;
    CPPUNIT_ASSERT(0 == 0);
  }
  
  void testMeshFunction()
  {
    // Write and read mesh function to/from file
    UnitSquare mesh(1, 1);
    MeshFunction<int> f(mesh, 0);
    f.set(0, 2);
    f.set(1, 4);
    f.set(2, 6);
    f.set(3, 8);
    File file("meshfunction.xml");
    file << f;
    MeshFunction<int> g(mesh, 0);
    file >> g;
    for (VertexIterator v(mesh); !v.end(); ++v)
      CPPUNIT_ASSERT(f(*v) == g(*v));
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

CPPUNIT_TEST_SUITE_REGISTRATION(SimpleShapes);
CPPUNIT_TEST_SUITE_REGISTRATION(MeshRefinement);
CPPUNIT_TEST_SUITE_REGISTRATION(MeshIterators);
CPPUNIT_TEST_SUITE_REGISTRATION(BoundaryExtraction);
CPPUNIT_TEST_SUITE_REGISTRATION(MeshFunctions);
CPPUNIT_TEST_SUITE_REGISTRATION(InputOutput);
CPPUNIT_TEST_SUITE_REGISTRATION(PyCCInterface);

int main()
{
  DOLFIN_TEST;
}
