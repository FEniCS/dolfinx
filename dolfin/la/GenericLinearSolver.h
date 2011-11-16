// Copyright (C) 2008-2010 Garth N. Wells
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
// Modified by Anders Logg 2009-2011
//
// First added:  2008-08-26
// Last changed: 2011-11-11

#ifndef __GENERIC_LINEAR_SOLVER_H
#define __GENERIC_LINEAR_SOLVER_H

#include <boost/shared_ptr.hpp>
#include <dolfin/common/Variable.h>
#include <dolfin/log/log.h>

namespace dolfin
{

  // Forward declarations
  class GenericMatrix;
  class GenericVector;

  /// This class provides a general solver for linear systems Ax = b.

  class GenericLinearSolver : public Variable
  {
  public:

    /// Set operator (matrix)
    virtual void set_operator(const boost::shared_ptr<const GenericMatrix> A) = 0;

    /// Set operator (matrix) and preconditioner matrix
    virtual void set_operators(const boost::shared_ptr<const GenericMatrix> A,
                               const boost::shared_ptr<const GenericMatrix> P)
    {
      dolfin_error("GenericLinearSolver.h",
                   "set operator and preconditioner for linear solver",
                   "Not supported by current linear algebra backend");
    }

    /// Solve linear system Ax = b
    virtual uint solve(const GenericMatrix& A, GenericVector& x, const GenericVector& b)
    {
      dolfin_error("GenericLinearSolver.h",
                   "solve linear systtem",
                   "Not supported by current linear algebra backend. Consider using solve(x, b)");
      return 0;
    }

    /// Solve linear system Ax = b
    virtual uint solve(GenericVector& x, const GenericVector& b)
    {
      dolfin_error("GenericLinearSolver.h",
                   "solve linear systtem",
                   "Not supported by current linear algebra backend. Consider using solve(x, b)");
      return 0;
    }

  };

}

#endif
