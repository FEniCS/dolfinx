// Copyright (C) 2014 Garth N. Wells
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

#include <string>
#include "GenericLinearAlgebraFactory.h"
#include "GenericLinearOperator.h"
#include "GenericMatrix.h"
#include "GenericVector.h"
#include "VectorSpaceBasis.h"
#include "test_nullspace.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
bool dolfin::in_nullspace(const GenericLinearOperator& A,
                          const VectorSpaceBasis& basis, std::string type)
{
  // Tolerance (maybe this should be a parameter?)
  const double tol = 1.0e-7;

  // Get dimension, and return if basis is empty
  const std::size_t dim = basis.dim();
  if (dim == 0)
    return true;

  // Get factory and create vector for LHS
  dolfin_assert(basis[0]);
  GenericLinearAlgebraFactory& factory = basis[0]->factory();
  std::shared_ptr<GenericVector> y = factory.create_vector(basis[0]->mpi_comm());

  const GenericMatrix* _A = NULL;
  if (type == "right")
  {
    // Do nothing
  }
  else if (type == "left")
  {
    const GenericMatrix* _A = dynamic_cast<const GenericMatrix*>(&A);
    if (!_A)
    {
      dolfin_error("test_nullspace.cpp",
                   "calling is_nullspace(...)",
                   "Left nullspace can be tested for GenericMatrix only (not GenericLinearOperator)");
    }
  }
  else
  {
    dolfin_error("test_nullspace.cpp",
                 "calling is_nullspace(...)",
                 "Left nullspace can be tested for GenericMatrix only (not GenericLinearOperator)");
  }

  // Test nullspace
  for (std::size_t i = 0; i < dim; ++i)
  {
    std::shared_ptr<const GenericVector> x = basis[i];
    dolfin_assert(x);
    if (!_A)
      A.mult(*x, *y);
    else
      _A->transpmult(*x, *y);

    if (y->norm("l2") > tol)
      return false;
  }

  return true;
}
//-----------------------------------------------------------------------------
