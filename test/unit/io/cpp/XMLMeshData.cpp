// Copyright (C) 2013 Garth N. Wells
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
// First added:  2012-05-25
// Last changed:
//

#include <dolfin.h>
#include <dolfin/common/unittest.h>

using namespace dolfin;

class XMLMeshDataIO : public CppUnit::TestFixture
{
  CPPUNIT_TEST_SUITE(XMLMeshDataIO);
  CPPUNIT_TEST(test_write_read);
  CPPUNIT_TEST_SUITE_END();

public:

  void test_write_read()
  {
    // XML mesh output is not supported in parallel. Add test for
    // parallel with HDF5 when ready.
    if (dolfin::MPI::size(MPI_COMM_WORLD) == 1)
    {
      const std::size_t value = 10;
      {
        UnitSquareMesh mesh(2, 2);

        // Create some mesh data
        std::vector<std::size_t>& data0 = mesh.data().create_array("v", 0);
        data0.resize(mesh.num_entities(0), value);

        mesh.init(1);
        std::vector<std::size_t>& data1 = mesh.data().create_array("e", 1);
        data1.resize(mesh.num_entities(1), value);

        std::vector<std::size_t>& data2 = mesh.data().create_array("c", 2);
        data2.resize(mesh.num_entities(2), value);

        File file("mesh_data.xml");
        file << mesh;
      }

      {
        // Read mesh from file
        Mesh mesh("mesh_data.xml");

        // Access mesh data and check
        const std::vector<std::size_t>& data0 = mesh.data().array("v", 0);
        CPPUNIT_ASSERT(data0.size() == mesh.num_entities(0));
        CPPUNIT_ASSERT(data0[2] == value);
        const std::vector<std::size_t>& data1 = mesh.data().array("e", 1);
        CPPUNIT_ASSERT(data1.size() == mesh.num_entities(1));
        CPPUNIT_ASSERT(data1[2] == value);
        const std::vector<std::size_t>& data2 = mesh.data().array("c", 2);
        CPPUNIT_ASSERT(data2.size() == mesh.num_entities(2));
        CPPUNIT_ASSERT(data2[2] == value);
      }
    }
  }
};


CPPUNIT_TEST_SUITE_REGISTRATION(XMLMeshDataIO);

int main()
{
  DOLFIN_TEST;
}
