// Copyright (C) 2004-2011 Anders Logg and Garth N. Wells
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
// Modified by Garth N. Wells, 2006-2010.
// Modified by Ola Skavhaug 2008.
//
// First added:  2004-06-19
// Last changed: 2011-10-06

#ifndef __LINEAR_SOLVER_H
#define __LINEAR_SOLVER_H

#include <string>
#include <boost/scoped_ptr.hpp>
#include <dolfin/common/types.h>
#include "GenericLinearSolver.h"

namespace dolfin
{

  class GenericMatrix;
  class GenericVector;
  class LUSolver;
  class KrylovSolver;

  /// This class provides a general solver for linear systems Ax = b.

  class LinearSolver : public GenericLinearSolver
  {
  public:

    /// Create linear solver
    LinearSolver(std::string method = "default",
                 std::string preconditioner = "default");

    /// Destructor
    ~LinearSolver();

    /// Set the operator (matrix)
    void set_operator(const boost::shared_ptr<const GenericMatrix> A);

    /// Set the operator (matrix) and preconitioner matrix
    void set_operators(const boost::shared_ptr<const GenericMatrix> A,
                       const boost::shared_ptr<const GenericMatrix> P);

    /// Solve linear system Ax = b
    uint solve(const GenericMatrix& A, GenericVector& x, const GenericVector& b);

    /// Solve linear system Ax = b
    uint solve(GenericVector& x, const GenericVector& b);

    /// Default parameter values
    static Parameters default_parameters()
    {
      Parameters p("linear_solver");
      return p;
    }

  private:

    // Friends
    friend class LUSolver;
    friend class KrylovSolver;

    // Check whether string is contained in list
    static bool
    in_list(const std::string& method,
            const std::vector<std::pair<std::string, std::string> > methods);

    // Solver
    boost::scoped_ptr<GenericLinearSolver> solver;

  };

}

#endif
