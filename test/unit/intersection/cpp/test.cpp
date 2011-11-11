// Copyright (C) 2011  André Massing
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
// Modified by André Massing, 2011
//
// First added:  2011-10-04
// Last changed: 2011-11-10
//
// Unit test for the IntersectionOperator

#include <dolfin.h>
#include <dolfin/common/unittest.h>
#include <dolfin/intersection/cgal_includes.h>

#include <vector>
#include <algorithm>

using namespace dolfin;
using dolfin::uint;
  
  template <uint dim0, uint dim1> 
  void testEntityEntityIntersection(const Mesh & mesh)
  {
    //Compute incidences
    mesh.init(dim0,dim1);
    mesh.init(dim1,dim0);
    mesh.init(0,dim0);

    uint label = 1;
    //Default is to mark all entities
    MeshFunction<uint> labels(mesh,dim0,label);
    IntersectionOperator io(labels, label, "ExactPredicates");

    // Iterator over all entities and compute self-intersection
    // Should be same as looking up mesh incidences
    // as we use an exact kernel
    for (MeshEntityIterator entity(mesh,dim1); !entity.end(); ++entity)
    {
      // Compute intersection
      std::vector<uint> ids_result;
      io.all_intersected_entities(*entity,ids_result);
      //sort them but they are already unique.
      std::sort(ids_result.begin(),ids_result.end());

      // Compute intersections via vertices and connectivity
      // information. Two entities of the same only intersect
      // if they share at least one verte
      std::vector<uint> ids_result_2;
      if (dim1 > 0)
      {
	for (VertexIterator vertex(*entity); !vertex.end(); ++vertex)
	{
	  uint num_ent = vertex->num_entities(dim0);
	  const uint * entities = vertex->entities(dim0);
	  for (uint i = 0; i < num_ent; ++i)
	    ids_result_2.push_back(entities[i]);
	}
      }
      // If we have a vertex simply take the incidences.
      else if (dim0 > 0)
      {
	uint num_ent = entity->num_entities(dim0);
	const uint * entities = entity->entities(dim0);
	for (uint i = 0; i < num_ent; ++i)
	  ids_result_2.push_back(entities[i]);
      }
      else
      {
	ids_result_2.push_back(entity->index());
      }
      //Sorting and removing duplicates
      std::sort(ids_result_2.begin(),ids_result_2.end());
      std::vector<uint>::iterator it = std::unique(ids_result_2.begin(),ids_result_2.end());
      ids_result_2.resize(it - ids_result_2.begin());

      // Check against mesh incidences
      uint last = ids_result.size() - 1;
      CPPUNIT_ASSERT(ids_result.size() == ids_result_2.size()); 
      CPPUNIT_ASSERT(ids_result[0] == ids_result_2[0]); 
      CPPUNIT_ASSERT(ids_result[last] == ids_result_2[last]); 
    }
  }

class IntersectionOperator3D : public CppUnit::TestFixture
{
  CPPUNIT_TEST_SUITE(IntersectionOperator3D);

  CPPUNIT_TEST(testCellCellIntersection);
  CPPUNIT_TEST(testCellFacetIntersection);
  //Intersection betweenn tets and segments does not work yet
  //CPPUNIT_TEST(testCellEdgeIntersection);
  CPPUNIT_TEST(testCellVertexIntersection);

  CPPUNIT_TEST(testFacetFacetIntersection);
  CPPUNIT_TEST(testFacetEdgeIntersection);
  CPPUNIT_TEST(testFacetVertexIntersection);

  CPPUNIT_TEST(testEdgeEdgeIntersection);
  CPPUNIT_TEST(testEdgeVertexIntersection);
  CPPUNIT_TEST(testVertexVertexIntersection);

  CPPUNIT_TEST_SUITE_END();

public:

  void testCellCellIntersection() 
  { 
    uint N = 3;
    UnitCube mesh(N,N,N);
    testEntityEntityIntersection<3,3>(mesh);
  }

  void testCellFacetIntersection() 
  {
    uint N = 3;
    UnitCube mesh(N,N,N);
    testEntityEntityIntersection<3,2>(mesh);
  }

  void testCellEdgeIntersection() 
  {
    uint N = 3;
    UnitCube mesh(N,N,N);
    testEntityEntityIntersection<3,1>(mesh);
  }

  void testCellVertexIntersection() 
  {
    uint N = 3;
    UnitCube mesh(N,N,N);
    testEntityEntityIntersection<3,0>(mesh);
  }

  void testFacetFacetIntersection()
  {
    uint N = 3;
    UnitCube mesh(N,N,N);
    testEntityEntityIntersection<2,2>(mesh);
  }

  void testFacetEdgeIntersection() 
  {
    uint N = 3;
    UnitCube mesh(N,N,N);
    testEntityEntityIntersection<2,1>(mesh);
  }

  void testFacetVertexIntersection()
  {
    uint N = 3;
    UnitCube mesh(N,N,N);
    testEntityEntityIntersection<2,0>(mesh);
  }

  void testEdgeEdgeIntersection() 
  {
    uint N = 3;
    UnitCube mesh(N,N,N);
    testEntityEntityIntersection<1,1>(mesh);
  }

  void testEdgeVertexIntersection()
  {
    uint N = 3;
    UnitCube mesh(N,N,N);
    testEntityEntityIntersection<1,0>(mesh);
  }

  void testVertexVertexIntersection() 
  { 
    uint N = 3;
    UnitCube mesh(N,N,N);
    testEntityEntityIntersection<0,0>(mesh);
  }

};

class IntersectionOperator2D : public CppUnit::TestFixture
{
  CPPUNIT_TEST_SUITE(IntersectionOperator2D);

  CPPUNIT_TEST(testCellCellIntersection);
  CPPUNIT_TEST(testCellEdgeIntersection);
  CPPUNIT_TEST(testCellVertexIntersection);

  CPPUNIT_TEST(testEdgeEdgeIntersection);
  CPPUNIT_TEST(testEdgeVertexIntersection);

  CPPUNIT_TEST(testVertexVertexIntersection);

  CPPUNIT_TEST_SUITE_END();

public:

  void testCellCellIntersection() 
  { 
    uint N = 6;
    UnitSquare mesh(N,N);
    testEntityEntityIntersection<2,2>(mesh);
  }

  void testCellEdgeIntersection() 
  {
    uint N = 6;
    UnitSquare mesh(N,N);
    testEntityEntityIntersection<2,1>(mesh);
  }

  void testCellVertexIntersection() 
  {
    uint N = 6;
    UnitSquare mesh(N,N);
    testEntityEntityIntersection<2,0>(mesh);
  }

  void testEdgeEdgeIntersection() 
  {
    uint N = 6;
    UnitSquare mesh(N,N);
    testEntityEntityIntersection<1,1>(mesh);
  }

  void testEdgeVertexIntersection()
  {
    uint N = 6;
    UnitSquare mesh(N,N);
    testEntityEntityIntersection<1,0>(mesh);
  }

  void testVertexVertexIntersection() 
  { 
    uint N = 6;
    UnitSquare mesh(N,N);
    testEntityEntityIntersection<0,0>(mesh);
  }

};

class IntersectionOperator1D : public CppUnit::TestFixture
{
  CPPUNIT_TEST_SUITE(IntersectionOperator1D);

  CPPUNIT_TEST(testCellCellIntersection);
  CPPUNIT_TEST(testCellVertexIntersection);

  CPPUNIT_TEST(testVertexVertexIntersection);

  CPPUNIT_TEST_SUITE_END();

public:

  void testCellCellIntersection() 
  { 
    uint N = 10;
    UnitInterval mesh(N);
    testEntityEntityIntersection<1,1>(mesh);
  }

  void testCellVertexIntersection() 
  {
    uint N = 10;
    UnitInterval mesh(N);
    testEntityEntityIntersection<1,0>(mesh);
  }

  void testVertexVertexIntersection() 
  { 
    uint N = 10;
    UnitInterval mesh(N);
    testEntityEntityIntersection<0,0>(mesh);
  }

};

int main()
{
  // FIXME: The following tests break probably in parallel
  if (dolfin::MPI::num_processes() == 1)
  {
    CPPUNIT_TEST_SUITE_REGISTRATION(IntersectionOperator3D);
    CPPUNIT_TEST_SUITE_REGISTRATION(IntersectionOperator2D);
    CPPUNIT_TEST_SUITE_REGISTRATION(IntersectionOperator1D);
  }

  DOLFIN_TEST;
  
}
