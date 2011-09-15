// Copyright (C) 2007-2008 Anders Logg
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
// Modified by Ola Skavhaug 2008.
// Modified by Garth N. Wells 2011.
//
// First added:  2007-04-30
// Last changed: 2011-09-15

#include <boost/scoped_ptr.hpp>

#include <dolfin/common/Timer.h>
#include "GenericMatrix.h"
#include "GenericVector.h"
#include "LinearAlgebraFactory.h"
#include "LinearSolver.h"
#include "solve.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
dolfin::uint dolfin::solve(const GenericMatrix& A, GenericVector& x, const GenericVector& b,
                   std::string solver_type, std::string pc_type)
{
  Timer timer("Solving linear system");
  LinearSolver solver(solver_type, pc_type);
  return solver.solve(A, x, b);
}
//-----------------------------------------------------------------------------
double dolfin::residual(const GenericMatrix& A, const GenericVector& x,
                        const GenericVector& b)
{
  boost::scoped_ptr<GenericVector> y(A.factory().create_vector());
  A.mult(x, *y);
  *y -= b;
  return y->norm("l2");
}
//-----------------------------------------------------------------------------
double dolfin::normalize(GenericVector& x, std::string normalization_type)
{
  double c = 0.0;
  if (normalization_type == "l2")
  {
    c = x.norm("l2");
    x /= c;
  }
  else if (normalization_type == "average")
  {
    boost::scoped_ptr<GenericVector> y(x.factory().create_vector());
    y->resize(x.size());
    (*y) = 1.0 / static_cast<double>(x.size());
    c = x.inner(*y);
    (*y) = c;
    x -= (*y);
  }
  else
    error("Unknown normalization type.");

  return c;
}
//-----------------------------------------------------------------------------

