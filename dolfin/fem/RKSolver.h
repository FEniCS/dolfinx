// Copyright (C) 2013 Johan Hake
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
// First added:  2013-02-15
// Last changed: 2013-02-25

#ifndef __RKSOLVER_H
#define __RKSOLVER_H

#include <vector>
#include <boost/shared_ptr.hpp>

#include <dolfin/function/FunctionAXPY.h>

namespace dolfin
{

  /// This class is a time integrator for general Runge Kutta problems

  // Forward declarations
  class ButcherScheme;

  class RKSolver
  {
  public:

    /// Constructor
    /// FIXME: Include version where one can pass a Solver
    RKSolver(boost::shared_ptr<ButcherScheme> scheme);

    /// Step solver with time step dt
    void step(double dt);

    /// Step solver with time step dt
    void step_interval(double t0, double t1);

  private:

    // The ButcherScheme
    boost::shared_ptr<ButcherScheme> _scheme;

  };

}

#endif
