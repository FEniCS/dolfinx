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
// Last changed: 2013-03-04

#ifndef __RKSOLVER_H
#define __RKSOLVER_H

#include <vector>
#include <boost/shared_ptr.hpp>

#include <dolfin/function/FunctionAXPY.h>

#include "Assembler.h"

namespace dolfin
{

  /// This class is a time integrator for general Runge Kutta forms,
  /// which only includes Point integrals with piecewise linear test
  /// functions. Such problems are disconnected at the vertices and
  /// can therefore be solved locally.

  // Forward declarations
  class ButcherScheme;
  class UFC;

  class PointIntegralSolver
  {
  public:

    /// Constructor
    /// FIXME: Include version where one can pass a Solver and/or Parameters
    PointIntegralSolver(boost::shared_ptr<ButcherScheme> scheme);

    /// Step solver with time step dt
    void step(double dt);

    /// Step solver an interval using dt as time step
    void step_interval(double t0, double t1, double dt);

  private:

    // Check the forms making sure they only include piecewise linear
    // test functions
    void _check_forms();

    // Build map between vertices, cells and the correspondning local vertex
    // and initialize UFC data for each form
    void _init();

    // The ButcherScheme
    boost::shared_ptr<ButcherScheme> _scheme;

    // Assembler for explicit stages
    Assembler _assembler;

    // Vertex map between vertices, cells and corresponding local vertex
    std::vector<std::pair<std::size_t, unsigned int> > _vertex_map;

    // UFC objects, one for each form
    std::vector<std::vector<boost::shared_ptr<UFC> > > _ufcs;

    // Solution coefficient index in form
    std::vector<std::vector<int> > _coefficient_index;

  };

}

#endif
