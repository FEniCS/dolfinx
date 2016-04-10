// Copyright (C) 2007-2009 Garth N. Wells
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
// Modified by Ola Skavhaug, 2008.
// Modified by Anders Logg, 2008.
//
// First added:  2007-07-03
// Last changed: 2014-05-27

#ifndef __KRYLOV_SOLVER_H
#define __KRYLOV_SOLVER_H

#include <string>
#include <vector>
#include <memory>
#include "GenericLinearSolver.h"
#include <dolfin/common/MPI.h>

namespace dolfin
{

  class GenericLinearOperator;
  class GenericVector;
  class VectorSpaceBasis;

  /// This class defines an interface for a Krylov solver. The
  /// appropriate solver is chosen on the basis of the matrix/vector
  /// type.

  class KrylovSolver : public GenericLinearSolver
  {
  public:

    /// Constructor
    KrylovSolver(MPI_Comm comm,
                 std::string method="default",
                 std::string preconditioner="default");

    /// Constructor
    KrylovSolver(std::string method="default",
                 std::string preconditioner="default");

    /// Constructor
    KrylovSolver(MPI_Comm comm,
                 std::shared_ptr<const GenericLinearOperator> A,
                 std::string method="default",
                 std::string preconditioner="default");

    /// Constructor
    KrylovSolver(std::shared_ptr<const GenericLinearOperator> A,
                 std::string method="default",
                 std::string preconditioner="default");

    /// Destructor
    ~KrylovSolver();

    /// Set operator (matrix)
    void set_operator(std::shared_ptr<const GenericLinearOperator> A);

    /// Set operator (matrix) and preconditioner matrix
    void set_operators(std::shared_ptr<const GenericLinearOperator> A,
                       std::shared_ptr<const GenericLinearOperator> P);

    /// Solve linear system Ax = b
    std::size_t solve(GenericVector& x, const GenericVector& b);

    /// Solve linear system Ax = b
    std::size_t solve(const GenericLinearOperator& A,
                      GenericVector& x, const GenericVector& b);

    /// Default parameter values
    static Parameters default_parameters();

    /// Update solver parameters (pass parameters down to wrapped implementation)
    virtual void update_parameters(const Parameters& parameters)
    {
      this->parameters.update(parameters);
      solver->parameters.update(parameters);
    }

    // FIXME: This should not be needed. Need to cleanup linear solver
    // name jungle: default, lu, iterative, direct, krylov, etc
    /// Return parameter type: "krylov_solver" or "lu_solver"
    std::string parameter_type() const
    {
      return "krylov_solver";
    }

  private:

    // Initialize solver
    void init(std::string method, std::string preconditioner, MPI_Comm comm);

    // Solver
    std::shared_ptr<GenericLinearSolver> solver;

  };
}

#endif
