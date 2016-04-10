// Copyright (C) 2007-2010 Garth N. Wells
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
// Modified by Ola Skavhaug 2008
// Modified by Dag Lindbo 2008
// Modified by Anders Logg 2008-2012
// Modified by Kent-Andre Mardal 2008
//
// First added:  2007-07-03
// Last changed: 2014-05-27

#ifndef __LU_SOLVER_H
#define __LU_SOLVER_H

#include <string>
#include <memory>
#include "GenericLUSolver.h"
#include <dolfin/common/MPI.h>

namespace dolfin
{

  // Forward declarations
  class GenericLinearOperator;
  class GenericVector;

  /// LU solver for the built-in LA backends.

  class LUSolver : public GenericLUSolver
  {
  public:

    /// Constructor
    LUSolver(MPI_Comm comm, std::string method= "default");

    /// Constructor
    LUSolver(std::string method= "default");

    /// Constructor
    LUSolver(MPI_Comm comm,
             std::shared_ptr<const GenericLinearOperator> A,
             std::string method="default");

    /// Destructor
    ~LUSolver();

    /// Set operator (matrix)
    void set_operator(std::shared_ptr<const GenericLinearOperator> A);

    /// Solve linear system Ax = b
    std::size_t solve(GenericVector& x, const GenericVector& b);

    /// Solve linear system A^Tx = b
    std::size_t solve_transpose(GenericVector& x, const GenericVector& b);

    /// Solve linear system
    std::size_t solve(const GenericLinearOperator& A, GenericVector& x,
                      const GenericVector& b);

    /// Solve linear system
    std::size_t solve_transpose(const GenericLinearOperator& A,
                                GenericVector& x, const GenericVector& b);

    /// Default parameter values
    static Parameters default_parameters()
    {
      Parameters p("lu_solver");
      p.add("report", true);
      p.add("verbose", false);
      p.add("symmetric", false);
      p.add("same_nonzero_pattern", false);   // deprecated
      p.add("reuse_factorization", false);   // deprecated
      return p;
    }

    /// Update solver parameters (pass parameters down to wrapped
    /// implementation)
    virtual void update_parameters(const Parameters& parameters)
    {
      this->parameters.update(parameters);
      solver->parameters.update(parameters);
    }

  private:

    // Initialize solver
    void init(MPI_Comm comm, std::string method);

    // Solver
    std::shared_ptr<GenericLinearSolver> solver;

  };
}

#endif
