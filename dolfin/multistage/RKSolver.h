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
// Last changed: 2014-03-05

#ifndef __RKSOLVER_H
#define __RKSOLVER_H

#include <vector>
#include <memory>
#include <dolfin/fem/Assembler.h>
#include <dolfin/la/GenericVector.h>

namespace dolfin
{
  // Forward declarations
  class MultiStageScheme;

  /// This class is a time integrator for general Runge Kutta problems

  class RKSolver
  {
  public:

    /// Constructor
    /// FIXME: Include version where one can pass a Solver and/or Parameters
    explicit RKSolver(std::shared_ptr<MultiStageScheme> scheme);

    /// Step solver with time step dt
    void step(double dt);

    /// Step solver an interval using dt as time step
    void step_interval(double t0, double t1, double dt);

    /// Return the MultiStageScheme
    std::shared_ptr<MultiStageScheme> scheme() const
    {return _scheme;}

  private:

    // The MultiStageScheme
    std::shared_ptr<MultiStageScheme> _scheme;

    // Temp vector for final stage
    // FIXME: Add this as a Function called previous step or something
    std::shared_ptr<GenericVector> _tmp;

    // Assembler for explicit stages
    Assembler _assembler;

  };

}

#endif
