/* -*- C -*- */
// Copyright (C) 2007-2012 Johan Hake
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
// Modified by Anders logg, 2009.
// Modified by Garth N. Wells, 2009.
//
// First added:  2007-11-25
// Last changed: 2012-01-18

//-----------------------------------------------------------------------------
// Declare shared_ptr stored types
//-----------------------------------------------------------------------------

%shared_ptr(dolfin::Hierarchical<dolfin::Form>)
%shared_ptr(dolfin::GenericDofMap)
%shared_ptr(dolfin::DofMap)
%shared_ptr(dolfin::Form)
%shared_ptr(dolfin::FiniteElement)
%shared_ptr(dolfin::BasisFunction)

%shared_ptr(dolfin::Hierarchical<dolfin::LinearVariationalProblem>)
%shared_ptr(dolfin::Hierarchical<dolfin::NonlinearVariationalProblem>)
%shared_ptr(dolfin::LinearVariationalProblem)
%shared_ptr(dolfin::NonlinearVariationalProblem)
%shared_ptr(dolfin::LinearVariationalSolver)
%shared_ptr(dolfin::NonlinearVariationalSolver)
%shared_ptr(dolfin::VariationalProblem)

%shared_ptr(dolfin::BoundaryCondition)
%shared_ptr(dolfin::Hierarchical<dolfin::DirichletBC>)
%shared_ptr(dolfin::DirichletBC)
%shared_ptr(dolfin::PeriodicBC)

