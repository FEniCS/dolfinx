// Copyright (C) 2008-2011 Anders Logg and Garth N. Wells
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
// Modified by Marie E. Rognes, 2011.
//
// First added:  2011-01-14 (2008-12-26 as VariationalProblem)
// Last changed: 2011-10-20

#ifndef __LINEAR_VARIATIONAL_SOLVER_H
#define __LINEAR_VARIATIONAL_SOLVER_H

#include <dolfin/common/Variable.h>
#include <dolfin/la/LUSolver.h>
#include <dolfin/la/KrylovSolver.h>

namespace dolfin
{

  // Forward declarations
  class LinearVariationalProblem;

  /// This class implements a solver for linear variational problems.

  class LinearVariationalSolver : public Variable
  {
  public:

    /// Create linear variational solver for given problem
    explicit LinearVariationalSolver(std::shared_ptr<LinearVariationalProblem> problem);

    /// Solve variational problem
    void solve();

    /// Default parameter values
    static Parameters default_parameters()
    {
      Parameters p("linear_variational_solver");

      p.add("linear_solver", "default");
      p.add("preconditioner", "default");
      p.add("symmetric", false);

      p.add("print_rhs", false);
      p.add("print_matrix", false);

      p.add(LUSolver::default_parameters());
      p.add(KrylovSolver::default_parameters());

      return p;
    }

  private:

    // The linear problem
    std::shared_ptr<LinearVariationalProblem> _problem;

  };

}

#endif
