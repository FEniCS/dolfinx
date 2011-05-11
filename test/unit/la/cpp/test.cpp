// Copyright (C) 2007 Johan Hake
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
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with DOLFIN.  If not, see <http://www.gnu.org/licenses/>.
//
// First added:  2008-09-30
// Last changed: 2008-09-30
//
// Unit tests linear algebra interface

// TODO: Implement!

#include <dolfin.h>
#include <dolfin/common/unittest.h>

using namespace dolfin;

class Default : public CppUnit::TestFixture
{
  CPPUNIT_TEST_SUITE(Default);
  CPPUNIT_TEST(testMatrix);
  CPPUNIT_TEST_SUITE_END();

public: 

  void testMatrix(){}
   
};

CPPUNIT_TEST_SUITE_REGISTRATION(Default);

int main()
{
  DOLFIN_TEST;
}
