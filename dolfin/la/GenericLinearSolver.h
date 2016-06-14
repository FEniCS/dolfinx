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
// Modified by Anders Logg 2009-2013
//
// First added:  2008-08-26
// Last changed: 2014-05-27

#ifndef __GENERIC_LINEAR_SOLVER_H
#define __GENERIC_LINEAR_SOLVER_H

#include <vector>
#include <memory>
#include <dolfin/common/Variable.h>
#include <dolfin/log/log.h>

namespace dolfin
{

  // Forward declarations
  class GenericLinearOperator;
  class GenericMatrix;
  class GenericVector;
  class VectorSpaceBasis;

  /// This class provides a general solver for linear systems Ax = b.

  class GenericLinearSolver : public Variable
  {
  public:

    /// Set operator (matrix)
    virtual void
      set_operator(std::shared_ptr<const GenericLinearOperator> A) = 0;

    /// Set operator (matrix) and preconditioner matrix
    virtual void
      set_operators(std::shared_ptr<const GenericLinearOperator> A,
                    std::shared_ptr<const GenericLinearOperator> P)
    {
      dolfin_error("GenericLinearSolver.h",
                   "set operator and preconditioner for linear solver",
                   "Not supported by current linear algebra backend");
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

    /// Solve linear system A^Tx = b
    virtual std::size_t solve_transpose(const GenericLinearOperator& A,
                                        GenericVector& x,
                                        const GenericVector& b)
    {
      dolfin_error("GenericLinearSolver.h",
                   "solve linear system transpose",
                   "Not supported by current linear algebra backend. Consider using solve_transpose(x, b)");
      return 0;
    }

    /// Solve linear system A^Tx = b
    virtual std::size_t solve_transpose(GenericVector& x,
                                        const GenericVector& b)
    {
      dolfin_error("GenericLinearSolver.h",
                   "solve linear system transpose",
                   "Not supported by current linear algebra backend. Consider using solve_transpose(x, b)");
      return 0;
    }

    // FIXME: This should not be needed. Need to cleanup linear solver
    // name jungle: default, lu, iterative, direct, krylov, etc
    /// Return parameter type: "krylov_solver" or "lu_solver"
    virtual std::string parameter_type() const
    {
      return "default";
    }

    /// Update solver parameters (useful for LinearSolver wrapper)
    virtual void update_parameters(const Parameters& parameters)
    {
      this->parameters.update(parameters);
    }

  protected:

    // Developer note: The functions here provide similar functionality
    // as the as_type functions in the LinearAlgebraObject base class. The
    // difference is that they specifically complain that a matrix is
    // required, which gives a user a more informative error message
    // from solvers that don't support matrix-free representation of
    // linear operators.

    // Down-cast GenericLinearOperator to GenericMatrix when an actual
    // matrix is required, not only a linear operator. This is the
    // const reference version of the down-cast.
    static const GenericMatrix& require_matrix(const GenericLinearOperator& A);

    // Down-cast GenericLinearOperator to GenericMatrix when an actual
    // matrix is required, not only a linear operator. This is the
    // const reference version of the down-cast.
    static std::shared_ptr<const GenericMatrix>
    require_matrix(std::shared_ptr<const GenericLinearOperator> A);

  };

}

#endif
