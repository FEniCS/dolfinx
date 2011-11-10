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


class Intersection3D : public CppUnit::TestFixture
{
  CPPUNIT_TEST_SUITE(Intersection3D);

  CPPUNIT_TEST(testCellCellIntersection);
  CPPUNIT_TEST(testCellFacetIntersection);
  //Intersection betweenn tets and segments does not work yet
  //CPPUNIT_TEST(testCellEdgeIntersection);
  CPPUNIT_TEST(testCellVertexIntersection);

  CPPUNIT_TEST(testFacetFacetIntersection);
//  CPPUNIT_TEST(testFacetEdgeIntersection);
  CPPUNIT_TEST(testFacetVertexIntersection);

  CPPUNIT_TEST(testEdgeEdgeIntersection);
  CPPUNIT_TEST(testEdgeVertexIntersection);
  CPPUNIT_TEST(testVertexVertexIntersectionNew);
  CPPUNIT_TEST(testVertexVertexIntersection);

  CPPUNIT_TEST_SUITE_END();

public:

  void testVertexVertexIntersectionNew()
  {
    UnitCube mesh(1,1,1);
    for (VertexIterator v1(mesh); !v1.end(); ++v1)
      for (VertexIterator v2(mesh); !v2.end(); ++v2)
      {
	if (v1->intersects(*v2))
	{
	  cout << "Vertex v1 = " << v1->index() << " intersects vertex v2 = " << v2->index() << endl;
	  info(v1->midpoint().str(true));
	  info(v2->midpoint().str(true));
	  assert(v1->index() == v2->index());
	}
      }

    info("Testing via Point Entity Intersections...");
    for (VertexIterator v1(mesh); !v1.end(); ++v1)
      for (VertexIterator v2(mesh); !v2.end(); ++v2)
	if (PrimitiveIntersector::do_intersect(*v1, v2->midpoint()))
	{
	  cout << "Vertex v1 = " << v1->index() << " intersects vertex v2 = " << v2->index() << endl;
	  info(v1->midpoint().str(true));
	  info(v2->midpoint().str(true));
	  assert(v1->index() == v2->index());
	}

    info("Testing via direct CGAL kernel use...");
    typedef EPICK::Point_3 Point_3;

    for (VertexIterator v1(mesh); !v1.end(); ++v1)
      for (VertexIterator v2(mesh); !v2.end(); ++v2)
      {
	Point_3 p1(v1->midpoint());
	Point_3 p2(v2->midpoint());
	if (p1 == p2)
	{
	  cout << "Vertex v1 = " << v1->index() << " intersects vertex v2 = " << v2->index() << endl;
	  info(v1->midpoint().str(true));
	  info(v2->midpoint().str(true));
	  assert(v1->index() == v2->index());
	}
      }
  }

  void testCellCellIntersection()
  {
    testEntityEntityIntersection<3,3>();
  }

  void testCellFacetIntersection()
  {
    testEntityEntityIntersection<3,2>();
  }

  void testCellEdgeIntersection()
  {
    testEntityEntityIntersection<3,1>();
  }

  void testCellVertexIntersection()
  {
    testEntityEntityIntersection<3,0>();
  }

  void testFacetFacetIntersection()
  {
    testEntityEntityIntersection<2,2>();
  }

  void testFacetEdgeIntersection()
  {
    testEntityEntityIntersection<2,1>();
  }

  void testFacetVertexIntersection()
  {
    testEntityEntityIntersection<2,0>();
  }

  void testEdgeEdgeIntersection()
  {
    testEntityEntityIntersection<1,1>();
  }

  void testEdgeVertexIntersection()
  {
    testEntityEntityIntersection<1,0>();
  }

  void testVertexVertexIntersection()
  {
    testEntityEntityIntersection<0,0>();
  }

  template <uint dim0, uint dim1> 
  void testEntityEntityIntersection()
  {
    cout <<"Run test with dim pair " << dim0 << " " << dim1 << endl;
    uint N = 1;
    UnitCube mesh(N,N,N);

    //Compute incidences
    mesh.init(dim0,dim1);
    mesh.init(dim1,dim0);

    MeshFunction<uint> labels(mesh,dim0,0);
    labels = 0;
//    IntersectionOperator io(mesh,"ExactPredicates");
    IntersectionOperator io(labels, 0, "ExactPredicates");

    // Iterator over all entities and compute self-intersection
    // Should be same as looking up mesh incidences
    // as we use an exact kernel
    for (MeshEntityIterator entity(mesh,dim1); !entity.end(); ++entity)
    {
      // Compute intersection
      std::vector<uint> ids_result;
      io.all_intersected_entities(*entity,ids_result);
      std::sort(ids_result.begin(),ids_result.end());
      cout << "--------------------------------------------------------------------------------" << endl;
      cout <<"Found " << ids_result.size() << " intersections" << endl;
      for (uint i = 0; i < ids_result.size(); ++i)
	cout <<ids_result[i] << " ";
      cout << endl;

      // Compute intersections via vertices and connectivity
      // information. Two entities of the same only intersect
      // if they share at least one vertex
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
      cout <<"Found " << ids_result_2.size() << " intersections via connectivity" << endl;
      cout <<endl;
      for (uint i = 0; i < ids_result_2.size(); ++i)
	cout <<ids_result_2[i] << " ";
      cout << endl;
      cout << "--------------------------------------------------------------------------------" << endl;

      // Check against mesh incidences
      CPPUNIT_ASSERT(ids_result.size() == ids_result_2.size()); 
      uint last = ids_result.size() - 1;
      CPPUNIT_ASSERT(ids_result[0] == ids_result_2[0]); 
      CPPUNIT_ASSERT(ids_result[last] == ids_result_2[last]); 
    }
  }

};

int main()
{
  // FIXME: The following test breaks in parallel
  if (dolfin::MPI::num_processes() == 1)
  {
    CPPUNIT_TEST_SUITE_REGISTRATION(Intersection3D);
  }

  DOLFIN_TEST;
  
}
