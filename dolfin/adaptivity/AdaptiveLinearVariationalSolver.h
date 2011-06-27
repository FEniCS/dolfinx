// Copyright (C) 2010 Marie E. Rognes
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
// Modified by Anders Logg, 2010-2011.
//
// First added:  2010-08-19
// Last changed: 2011-03-31

#ifndef __ADAPTIVE_LINEAR_VARIATIONAL_SOLVER_H
#define __ADAPTIVE_LINEAR_VARIATIONAL_SOLVER_H

#include <boost/shared_ptr.hpp>
#include <dolfin/fem/BoundaryCondition.h>

#include "GenericAdaptiveVariationalSolver.h"

namespace dolfin
{
  // Forward declarations
  class Function;
  class LinearVariationalProblem;
  class GoalFunctional;

  class AdaptiveLinearVariationalSolver
    : public GenericAdaptiveVariationalSolver
  {
  public:

    /// Create adaptive variational solver for given linear variaional
    /// problem
    AdaptiveLinearVariationalSolver(LinearVariationalProblem& problem);

    void solve(const double tol, GoalFunctional& M);

    boost::shared_ptr<const Function> solve_primal();

    std::vector<boost::shared_ptr<const BoundaryCondition> > extract_bcs() const;

  private:

    // The problem
    boost::shared_ptr<LinearVariationalProblem> problem;

  };

}

#endif
