// Copyright (C) 2007 Johan Hake.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2008-09-30
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
