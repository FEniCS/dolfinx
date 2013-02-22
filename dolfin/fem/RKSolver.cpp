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
// Last changed: 2013-02-22

#include <dolfin/function/Function.h>
#include <dolfin/function/Function.h>

#include "RKSolver.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
RKSolver::RKSolver(boost::shared_ptr<ButcherScheme> scheme) : 
  _scheme(scheme)
{}
//-----------------------------------------------------------------------------
void RKSolver::step(double dt)
{}
//-----------------------------------------------------------------------------
void RKSolver::step_interval(double t0, double t1)
{}
//-----------------------------------------------------------------------------
