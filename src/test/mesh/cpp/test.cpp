// Copyright (C) 2007 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2007-05-14
// Last changed: 2007-05-14
//
// Unit tests for the mesh library

#include <dolfin.h>
#include <dolfin/unittest.h>

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

CPPUNIT_TEST_SUITE_REGISTRATION(SimpleShapes);
CPPUNIT_TEST_SUITE_REGISTRATION(MeshRefinement);

int main()
{
  DOLFIN_TEST;
}
