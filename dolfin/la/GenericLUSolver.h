// Copyright (C) 2010 Garth N. Wells
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
// First added:  2010-07-11
// Last changed:

#ifndef __GENERIC_LU_SOLVER_H
#define __GENERIC_LU_SOLVER_H

#include <boost/shared_ptr.hpp>
#include <dolfin/common/Variable.h>
#include "GenericLinearSolver.h"

namespace dolfin
{

  /// Forward declarations
  class GenericVector;
  class GenericMatrix;

  /// This a base class for LU solvers

  class GenericLUSolver : public GenericLinearSolver
  {

  public:

    /// Set operator (matrix)
    virtual void set_operator(const GenericMatrix& A) = 0;

    /// Solve linear system Ax = b
    virtual uint solve(GenericVector& x, const GenericVector& b) = 0;

    /// Solve linear system Ax = b
    virtual uint solve(const GenericMatrix& A, GenericVector& x, const GenericVector& b)
    { error("solve(A, x, b) is not implemented. Consider trying solve(x, b)."); return 0; }

  };

}

#endif
