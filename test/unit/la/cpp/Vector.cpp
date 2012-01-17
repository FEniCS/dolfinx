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
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with DOLFIN. If not, see <http://www.gnu.org/licenses/>.
//
// First added:  2008-09-30
// Last changed: 2012-01-17
//
// Unit tests Selected methods for GenericVector

#include <dolfin.h>
#include <dolfin/common/unittest.h>

using namespace dolfin;

class TestVector : public CppUnit::TestFixture
{
  CPPUNIT_TEST_SUITE(TestVector);
  CPPUNIT_TEST(test_backends);
  CPPUNIT_TEST_SUITE_END();

public: 

  void test_backends()
  {

    // uBLAS
    parameters["linear_algebra_backend"] = "uBLAS";
    _test_operators();

    // FIXME: Outcommented STL backend to circumvent infinite loops as 
    // FIXME: seen on one buildbot
    // STL
    //parameters["linear_algebra_backend"] = "STL";
    //_test_operators();
    
    // PETSc
    #ifdef HAS_PETSC
    parameters["linear_algebra_backend"] = "PETSc";
    _test_operators();
    #endif
    
    // Epetra
    #ifdef HAS_EPETRA
    parameters["linear_algebra_backend"] = "Epetra";
    _test_operators();
    #endif
    
    // MTL4
    #ifdef HAS_MTL4
    parameters["linear_algebra_backend"] = "MTL4";
    _test_operators();
    #endif
    

  }

  void _test_operators()
  {
    Vector v(10), u(10);
    v = 0.0;
    u = 0.0;
    CPPUNIT_ASSERT(v.sum() == 0.0);
    
    // operator=(double a)
    v = 1.0;
    CPPUNIT_ASSERT(v.sum() == v.size());
    
    // operator=(const GenericVector& x)
    u = v;
    CPPUNIT_ASSERT(u.sum() == u.size());

    // operator+=(const GenericVector& x)
    u += v;
    CPPUNIT_ASSERT(u.sum() == 2*u.size());
    
    // operator-=(const GenericVector& x)
    u -= v;
    u -= v;
    CPPUNIT_ASSERT(u.sum() == 0.0);
    
    // operator*=(double a)
    v *= 5.0;
    CPPUNIT_ASSERT(v.sum() == v.size()*5.0);

    // operator/=(double a)
    v /= 2.0;
    CPPUNIT_ASSERT(v.sum() == 2.5*v.size());
    
    // operator*=(const GenericVector& x)
    u = 2.0;
    v*=u;
    CPPUNIT_ASSERT(v.sum() == v.size()*5.0);

  }
   
};

CPPUNIT_TEST_SUITE_REGISTRATION(TestVector);

int main()
{
  DOLFIN_TEST;
}
