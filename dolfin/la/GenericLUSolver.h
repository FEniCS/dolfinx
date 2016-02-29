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
// Modified by Anders Logg 2011-2012
//
// First added:  2010-07-11
// Last changed: 2012-08-20

#ifndef __GENERIC_LU_SOLVER_H
#define __GENERIC_LU_SOLVER_H

#include <memory>
#include <dolfin/common/Variable.h>
#include "GenericLinearSolver.h"

namespace dolfin
{

  /// Forward declarations
  class GenericVector;
  class GenericLinearOperator;

  /// This a base class for LU solvers

  class GenericLUSolver : public GenericLinearSolver
  {

  public:

    /// Set operator (matrix)
    virtual void
      set_operator(std::shared_ptr<const GenericLinearOperator> A) = 0;

    /// Solve linear system Ax = b
    virtual std::size_t solve(GenericVector& x, const GenericVector& b) = 0;

    /// Solve linear system Ax = b
    virtual std::size_t solve(const GenericLinearOperator& A,
                       GenericVector& x, const GenericVector& b)
    {
      dolfin_error("GenericLUSolver.h",
                   "solve linear system",
                   "Not supported by current linear algebra backend. Consider using solve(x, b)");
      return 0;
    }

    /// Solve linear system A^Tx = b
    virtual std::size_t solve_transpose(GenericVector& x, const GenericVector& b)
    {
      dolfin_error("GenericLUSolver.h",
                   "solve linear system",
                   "Not supported by current linear algebra backend.");
      return 0;
    }

    /// Solve linear system A^Tx = b
    virtual std::size_t solve_transpose(const GenericLinearOperator& A,
                       GenericVector& x, const GenericVector& b)
    {
      dolfin_error("GenericLUSolver.h",
                   "solve linear system",
                   "Not supported by current linear algebra backend.");
      return 0;
    }

    /// Return parameter type: "krylov_solver" or "lu_solver"
    virtual std::string parameter_type() const
    {
      return "lu_solver";
    }

  };

}

#endif
