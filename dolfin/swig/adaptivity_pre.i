/* -*- C -*- */
// Copyright (C) 2011 Marie E. Rognes
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
// First added:  2011-02-23
// Last changed: 2011-07-04

// ===========================================================================
// SWIG directives for the DOLFIN adaptivity kernel module (pre)
//
// The directives in this file are applied _before_ the header files of the
// modules has been loaded.
// ===========================================================================

namespace dolfin {
  class ErrorControl;
}

//-----------------------------------------------------------------------------
// Instantiate Hierarchical ErrorControl template class
//-----------------------------------------------------------------------------
%template (HierarchicalErrorControl) dolfin::Hierarchical<dolfin::ErrorControl>;

//-----------------------------------------------------------------------------
// Rename [] for SpecialFacetFunction -> _sub
//-----------------------------------------------------------------------------
%rename(_sub) dolfin::SpecialFacetFunction::operator[];

//-----------------------------------------------------------------------------
// Ignore solve methods of *Adaptive*VariationalSolver that take
// GoalFunctional as input
//-----------------------------------------------------------------------------
%ignore dolfin::GenericAdaptiveVariationalSolver::solve(const double tol,
                                                        GoalFunctional& M);
%ignore dolfin::AdaptiveLinearVariationalSolver::solve(const double tol,
                                                       GoalFunctional& M);
%ignore dolfin::AdaptiveNonlinearVariationalSolver::solve(const double tol,
                                                          GoalFunctional& M);

//-----------------------------------------------------------------------------
// Ignore GoalFunctional entirely
//-----------------------------------------------------------------------------
%ignore dolfin::GoalFunctional;
