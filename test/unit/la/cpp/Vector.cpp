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
// Last changed: 2012-08-21
//
// Unit tests Selected methods for GenericVector

#include <dolfin.h>
#include <dolfin/common/unittest.h>

using namespace dolfin;

class TestVector : public CppUnit::TestFixture
{
  CPPUNIT_TEST_SUITE(TestVector);
  CPPUNIT_TEST(test_backends);
  CPPUNIT_TEST(test_init);
  CPPUNIT_TEST_SUITE_END();

public:

  void test_backends()
  {
    // uBLAS
    parameters["linear_algebra_backend"] = "uBLAS";
    _test_operators(MPI_COMM_SELF);

    // FIXME: Outcommented STL backend to circumvent infinite loops as
    // FIXME: seen on one buildbot
    // STL
    //parameters["linear_algebra_backend"] = "STL";
    //_test_operators();

    // PETSc
    #ifdef HAS_PETSC
    parameters["linear_algebra_backend"] = "PETSc";
    _test_operators(MPI_COMM_WORLD);
    #endif

    // Epetra
    #ifdef HAS_EPETRA
    parameters["linear_algebra_backend"] = "Epetra";
    _test_operators(MPI_COMM_WORLD);
    #endif
  }

  void _test_operators(MPI_Comm comm)
  {
    Vector v(comm, 10), u(comm, 10);
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

  void test_init()
  {
    // Create local and distributed vector layouts
    const std::vector<std::size_t> dims(1, 203);

    // Create local vector layout
    TensorLayout layout_local(0, false);
    std::vector<std::pair<std::size_t, std::size_t> >
      local_range(1, std::make_pair(0, 203));
    layout_local.init(MPI_COMM_SELF, dims, 1, local_range);

    // Create distributed vector layout
    TensorLayout layout_distributed(0, false);
    std::vector<std::pair<std::size_t, std::size_t> >
      ownership_range(1, MPI::local_range(MPI_COMM_WORLD, 203));
    layout_distributed.init(MPI_COMM_WORLD, dims, 1, ownership_range);

    // Vector
    #ifdef HAS_PETSC
    parameters["linear_algebra_backend"] = "PETSc";
    {
      Vector x;
      x.init(layout_local);
      CPPUNIT_ASSERT(x.size() == 203);

      Vector y;
      y.init(layout_distributed);
      CPPUNIT_ASSERT(y.size() == 203);
    }
    #endif

    // uBLAS
    {
      uBLASVector x;
      x.init(layout_local);
      CPPUNIT_ASSERT(x.size() == 203);
    }

    // PETSc
    #ifdef HAS_PETSC
    {
      PETScVector x;
      x.init(layout_local);
      CPPUNIT_ASSERT(x.size() == 203);

      PETScVector y;
      y.init(layout_distributed);
      CPPUNIT_ASSERT(y.size() == 203);
    }
    #endif

    // Epetra
    #ifdef HAS_EPETRA
    {
      EpetraVector x;
      x.init(layout_local);
      CPPUNIT_ASSERT(x.size() == 203);

      EpetraVector y;
      y.init(layout_distributed);
      CPPUNIT_ASSERT(y.size() == 203);
    }
    #endif

  }

};

CPPUNIT_TEST_SUITE_REGISTRATION(TestVector);

int main()
{
  DOLFIN_TEST;
}
