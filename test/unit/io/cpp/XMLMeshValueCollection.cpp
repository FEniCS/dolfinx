// Copyright (C) 2007 Magnus Vikstrøm
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
// First added:  2007-05-29
// Last changed: 2012-01-12
//

#include <dolfin.h>
#include <dolfin/mesh/LocalMeshData.h>
#include <dolfin/common/unittest.h>

using namespace dolfin;

class MeshValueCollectionIO : public CppUnit::TestFixture
{
  CPPUNIT_TEST_SUITE(MeshValueCollectionIO);
  CPPUNIT_TEST(test_read);
  CPPUNIT_TEST_SUITE_END();

public:

  void test_read()
  {
    // Create mesh and read file
    auto mesh = std::make_shared<UnitCubeMesh>(5, 5, 5);
    MeshValueCollection<std::size_t>
      markers(mesh, "xml_value_collection_ref.xml");

    // Check size
    CPPUNIT_ASSERT(dolfin::MPI::sum(mesh->mpi_comm(), markers.size()) == 6);

    // Check sum of values
    const std::map<std::pair<std::size_t, std::size_t>, std::size_t>&
      values = markers.values();
    std::map<std::pair<std::size_t, std::size_t>,
             std::size_t>::const_iterator it;
    std::size_t sum = 0;
    for (it = values.begin(); it != values.end(); ++it)
      sum += it->second;
    CPPUNIT_ASSERT(dolfin::MPI::sum(mesh->mpi_comm(), sum) == 48);
  }

};


CPPUNIT_TEST_SUITE_REGISTRATION(MeshValueCollectionIO);

int main()
{
  DOLFIN_TEST;
}
