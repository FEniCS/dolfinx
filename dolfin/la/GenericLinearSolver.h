// Copyright (C) 2008-2013 Garth N. Wells
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
// Modified by Anders Logg 2009-2012
//
// First added:  2008-08-26
// Last changed: 2013-02-26

#ifndef __GENERIC_LINEAR_SOLVER_H
#define __GENERIC_LINEAR_SOLVER_H

#include <vector>
#include <boost/shared_ptr.hpp>
#include <dolfin/common/Variable.h>
#include <dolfin/log/log.h>
#include "GenericLinearOperator.h"
#include "GenericMatrix.h"

namespace dolfin
{

  // Forward declarations
  class GenericVector;

  /// This class provides a general solver for linear systems Ax = b.

  class GenericLinearSolver : public Variable
  {
  public:

    /// Set operator (matrix)
    virtual void set_operator(const boost::shared_ptr<const GenericLinearOperator> A) = 0;

    /// Set operator (matrix) and preconditioner matrix
    virtual void set_operators(const boost::shared_ptr<const GenericLinearOperator> A,
                               const boost::shared_ptr<const GenericLinearOperator> P)
    {
      dolfin_error("GenericLinearSolver.h",
                   "set operator and preconditioner for linear solver",
                   "Not supported by current linear algebra backend");
    }

    /// Set null space of the operator (matrix). This is used to solve
    /// singular systems
    virtual void set_nullspace(const std::vector<const GenericVector*> nullspace)
    {
      dolfin_error("GenericLinearSolver.h",
                   "set nullspace for operator",
                   "Not supported by current linear algebra solver backend");
    }

    /// Solve linear system Ax = b
    virtual std::size_t solve(const GenericLinearOperator& A, GenericVector& x,
                       const GenericVector& b)
    {
      dolfin_error("GenericLinearSolver.h",
                   "solve linear system",
                   "Not supported by current linear algebra backend. Consider using solve(x, b)");
      return 0;
    }

    /// Solve linear system Ax = b
    virtual std::size_t solve(GenericVector& x, const GenericVector& b)
    {
      dolfin_error("GenericLinearSolver.h",
                   "solve linear system",
                   "Not supported by current linear algebra backend. Consider using solve(x, b)");
      return 0;
    }

  protected:

    // Developer note: The functions here provide similar functionality
    // as the as_type functions in the LinearAlgebraObject base class. The
    // difference is that they specifically complain that a matrix is
    // required, which gives a user a more informative error message
    //from solvers that don't support matrix-free representation of
    //linear operators.

    // Down-cast GenericLinearOperator to GenericMatrix when an actual
    // matrix is required, not only a linear operator. This is the
    // const reference version of the down-cast.
    static const GenericMatrix& require_matrix(const GenericLinearOperator& A);

    // Down-cast GenericLinearOperator to GenericMatrix when an actual
    // matrix is required, not only a linear operator. This is the
    // const reference version of the down-cast.
    static boost::shared_ptr<const GenericMatrix>
    require_matrix(const boost::shared_ptr<const GenericLinearOperator> A);

  };

}

#endif
