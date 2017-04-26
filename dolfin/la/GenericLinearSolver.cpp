// Copyright (C) 2013 Garth N. Wells
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

#include "GenericLinearOperator.h"
#include "GenericLinearSolver.h"
#include "GenericMatrix.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
const GenericMatrix& GenericLinearSolver::require_matrix(const GenericLinearOperator& A)
{
  // Try to dynamic cast to a GenericMatrix
  try
  {
    return dynamic_cast<const GenericMatrix&>(A);
  }
  catch (std::exception& e)
  {
    dolfin_error("GenericLinearSolver.h",
                 "use linear operator as a matrix (real matrix required)",
                 "%s", e.what());
  }

  // Return something to keep the compiler happy, code will not be reached
  return dynamic_cast<const GenericMatrix&>(A);
}
//-----------------------------------------------------------------------------
std::shared_ptr<const GenericMatrix>
GenericLinearSolver::require_matrix(std::shared_ptr<const GenericLinearOperator> A)
{
  // Try to down cast shared pointer
  std::shared_ptr<const GenericMatrix> _matA
    = std::dynamic_pointer_cast<const GenericMatrix>(A);

  // Check results. Note the difference from the as_type functions
  // in LinearAlgebraObject in that we check the return value here
  // and throw an error if the cast fails.
  if (!_matA)
  {
    dolfin_error("GenericLinearSolver.h",
                 "use linear operator as a matrix (real matrix required)",
                 "Dynamic cast failed");
  }

  return _matA;
}
//-----------------------------------------------------------------------------
