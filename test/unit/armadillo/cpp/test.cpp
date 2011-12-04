// Copyright (C) 2011 Garth N. Wells
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
// First added:  2011-11-21
// Last changed:
//
// Unit tests for Armadillo. Main purpose it to test linkage to BLAS
// and LAPACK

#include <dolfin.h>
#include <dolfin/common/unittest.h>
#include <armadillo>

//using namespace dolfin;

class Default : public CppUnit::TestFixture
{
  CPPUNIT_TEST_SUITE(Default);
  CPPUNIT_TEST(test_square_solve);
  CPPUNIT_TEST(test_least_square_solve);
  CPPUNIT_TEST_SUITE_END();

public:

  void test_square_solve()
  {
    arma::mat A = arma::eye<arma::mat>(5, 5);
    arma::vec b = arma::randu<arma::vec>(5);
    arma::vec x = arma::solve(A, b);
    double norm = arma::norm(x - b, 2);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(0.0, norm, 1.0e-12);
  }

  void test_least_square_solve()
  {
    arma::mat A = arma::randu<arma::mat>(8, 6);
    arma::vec b = arma::randu<arma::vec>(8);
    arma::vec x;
    bool result = arma::solve(x, A, b);
    CPPUNIT_ASSERT(result);
  }

};

CPPUNIT_TEST_SUITE_REGISTRATION(Default);

int main()
{
  DOLFIN_TEST;
}
