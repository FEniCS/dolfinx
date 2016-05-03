/* -*- C -*- */
// Copyright (C) 2015 Johan Hake
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
// First added:  2015-02-17
// Last changed: 2015-02-17

//=============================================================================
// SWIG directives for the DOLFIN plot kernel module (pre)
//
// The directives in this file are applied _before_ the header files of the
// modules has been loaded.
//=============================================================================

//-----------------------------------------------------------------------------
// Ignores for MultiMesh
//-----------------------------------------------------------------------------
%ignore dolfin::plot(const MultiMesh&);
%rename (_plot_multimesh) dolfin::plot(std::shared_ptr<const MultiMesh>);

// Ignore reference versions of plot
%ignore dolfin::plot(const Variable&, std::string, std::string);
%ignore dolfin::plot(const Variable&, const Parameters&);
%ignore dolfin:: plot(const Expression&, const Mesh&, std::string, std::string);
%ignore dolfin:: plot(const Expression&, const Mesh&, std::string, std::string);
%ignore dolfin::plot(const Expression&, const Mesh&, const Parameters&);
