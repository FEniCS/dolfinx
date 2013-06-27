/* -*- C -*- */
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
// First added:  2013-04-23
// Last changed: 2013-04-24

//=============================================================================
// SWIG directives for the DOLFIN multistage kernel module (pre)
//
// The directives in this file are applied _before_ the header files of the
// modules has been loaded.
//=============================================================================

// Ignore access methods to MultiStageScheme as equivalent methods are added 
// to a wrapped Python class
%ignore dolfin::MultiStageScheme::stage_forms;
%ignore dolfin::MultiStageScheme::last_stage;
%ignore dolfin::MultiStageScheme::stage_solutions;
%ignore dolfin::MultiStageScheme::solution;
%ignore dolfin::MultiStageScheme::t;
%ignore dolfin::MultiStageScheme::dt;

%ignore dolfin::PointIntegralSolver::scheme;
%ignore dolfin::RKSolver::scheme;
