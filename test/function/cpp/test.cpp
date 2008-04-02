// Copyright (C) 2007 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2007-05-24
// Last changed: 2007-05-24
//
// Unit tests for the function library

#include <dolfin.h>
#include <dolfin/common/unittest.h>

using namespace dolfin;

class Foo : public CppUnit::TestFixture
{
  CPPUNIT_TEST_SUITE(Foo);
  CPPUNIT_TEST(testFoo);
  CPPUNIT_TEST_SUITE_END();

public: 

  void testFoo()
  {
    // No tests implemented
    CPPUNIT_ASSERT(0 == 0);
  }

};

CPPUNIT_TEST_SUITE_REGISTRATION(Foo);

int main()
{
  DOLFIN_TEST;
}
