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

class LocalMeshDataIO : public CppUnit::TestFixture
{
  CPPUNIT_TEST_SUITE(LocalMeshDataIO);
  CPPUNIT_TEST(testRead);
  CPPUNIT_TEST_SUITE_END();

public:

// FIXME: Not a proper unit test. When LocalMeshData has a public interface
// FIXME: we can expand on these
  void testRead()
  {
    // Create undirected graph with edges added out of order (should pass)
    File file("../snake.xml.gz");
    LocalMeshData localdata;
    file >> localdata;
  }
};


CPPUNIT_TEST_SUITE_REGISTRATION(LocalMeshDataIO);

int main()
{
  DOLFIN_TEST;
}
