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
// Last changed: 2011-10-25
//
// Unit test for the IntersectionOperator

#include <dolfin.h>
#include <dolfin/common/unittest.h>

#include <vector>

using namespace dolfin;
using dolfin::uint;


class Intersection3D : public CppUnit::TestFixture
{
  CPPUNIT_TEST_SUITE(Intersection3D);
  CPPUNIT_TEST(testCellCellIntersection);
//  CPPUNIT_TEST(testCellFacetIntersection);
//  CPPUNIT_TEST(testCellEdgeIntersection);
  CPPUNIT_TEST(testCellVertexIntersection);
  CPPUNIT_TEST_SUITE_END();

public:

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

  template <uint dim0, uint dim1> 
  void testEntityEntityIntersection()
  {
    cout <<"Run test with dim pair " << dim0 << " " << dim1 << endl;
    uint N = 5;
    UnitCube mesh(N,N,N);
    IntersectionOperator io(mesh,"ExactPredicates");
    
    //Compute incidences
    mesh.init(dim0,dim1);

    // Iterator over all entities and compute self-intersection
    // Should be same as looking up mesh incidences
    // as we use an exact kernel
   
    for (MeshEntityIterator entity(mesh,dim1); !entity.end(); ++entity)
    {
      // Compute intersection
      std::vector<uint> ids_result;
      io.all_intersected_entities(*entity,ids_result);
      cout << "--------------------------------------------------------------------------------" << endl;
      cout <<"Found " << ids_result.size() << " intersections" << endl;

      // Get mesh incidences
      uint num_ent = entity->num_entities(dim0);
      const uint * entities = entity->entities(dim0);
      cout <<"Found " << num_ent << " incidences" << endl;
      cout << "--------------------------------------------------------------------------------" << endl;

      // Check against mesh incidences
      for (uint i = 0; i < num_ent; ++i)
      {
	std::vector<uint>::iterator it = ids_result.begin();
	it = find(ids_result.begin(),ids_result.end(), entities[i]);
	CPPUNIT_ASSERT(it != ids_result.end()); 
      }
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
