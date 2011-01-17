// Copyright (C) 2007 Magnus Vikstrøm.
// Licensed under the GNU LGPL Version 2.1. 
//
// First added:  2007-05-29
// Last changed: 2007-05-29
//
// Unit tests for the graph library 

#include <dolfin.h>
#include <dolfin/mesh/LocalMeshData.h>
#include <dolfin/common/unittest.h>
#include <cstdlib>

using namespace dolfin;

class LocalMeshDataIO : public CppUnit::TestFixture
{
  CPPUNIT_TEST_SUITE(LocalMeshDataIO);
  CPPUNIT_TEST(testRead);
  CPPUNIT_TEST_SUITE_END();

public: 

  void testRead()
  {
    // Create undirected graph with edges added out of order (should pass)
    File file("../../../../data/meshes/snake.xml.gz");
    LocalMeshData localdata;
    file >> localdata;
  }
};
   

CPPUNIT_TEST_SUITE_REGISTRATION(LocalMeshDataIO);

int main()
{
  DOLFIN_TEST;
}
