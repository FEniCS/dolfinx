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
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with DOLFIN.  If not, see <http://www.gnu.org/licenses/>.
//
// Modified by Ola Skavhaug 2008.
//
// First added:  2007-04-30
// Last changed: 2008-08-19

#include <dolfin/common/Timer.h>
#include "LinearSolver.h"
#include "GenericMatrix.h"
#include "GenericVector.h"
#include "LinearAlgebraFactory.h"
#include "solve.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
void dolfin::solve(const GenericMatrix& A, GenericVector& x, const GenericVector& b,
                   std::string solver_type, std::string pc_type)
{
  Timer timer("Solving linear system");
  LinearSolver solver(solver_type, pc_type);
  solver.solve(A, x, b);
}
//-----------------------------------------------------------------------------
double dolfin::residual(const GenericMatrix& A, const GenericVector& x, 
                        const GenericVector& b)
{
  GenericVector* y = A.factory().create_vector();
  A.mult(x, *y);
  *y -= b;
  const double norm = y->norm("l2");
  delete y;
  return norm;
}
//-----------------------------------------------------------------------------
double dolfin::normalize(GenericVector& x, std::string normalization_type)
{
  if (normalization_type == "l2")
  {
    const double c = x.norm("l2");
    x /= c;
    return c;
  }
  else if (normalization_type == "average")
  {
    GenericVector* y = x.factory().create_vector();
    y->resize(x.size());
    (*y) = 1.0 / static_cast<double>(x.size());
    const double c = x.inner(*y);
    (*y) = c;
    x -= (*y);
    delete y;
    return c;
  }
  else
    error("Unknown normalization type.");

  return 0.0;
}
//-----------------------------------------------------------------------------

