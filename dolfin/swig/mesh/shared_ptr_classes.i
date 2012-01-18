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

%shared_ptr(dolfin::Hierarchical<dolfin::Mesh>)
%shared_ptr(dolfin::Mesh)
%shared_ptr(dolfin::BoundaryMesh)
%shared_ptr(dolfin::SubMesh)
%shared_ptr(dolfin::UnitTetrahedron)
%shared_ptr(dolfin::UnitCube)
%shared_ptr(dolfin::UnitInterval)
%shared_ptr(dolfin::Interval)
%shared_ptr(dolfin::UnitTriangle)
%shared_ptr(dolfin::UnitSquare)
%shared_ptr(dolfin::UnitCircle)
%shared_ptr(dolfin::Box)
%shared_ptr(dolfin::Rectangle)
%shared_ptr(dolfin::UnitSphere)

%shared_ptr(dolfin::SubDomain)
%shared_ptr(dolfin::DomainBoundary)

%shared_ptr(dolfin::LocalMeshData)
%shared_ptr(dolfin::MeshData)

// NOTE: Most of the MeshFunctions are declared sharepointers in
// NOTE: mesh_pre.i, mesh_post.i
%shared_ptr(dolfin::Hierarchical<dolfin::MeshFunction<dolfin::uint> >)
%shared_ptr(dolfin::MeshFunction<dolfin::uint>)

// FIXME: Do we need to declare dolfin::uint?
%shared_ptr(dolfin::CellFunction<dolfin::uint>)
%shared_ptr(dolfin::EdgeFunction<dolfin::uint>)
%shared_ptr(dolfin::FaceFunction<dolfin::uint>)
%shared_ptr(dolfin::FacetFunction<dolfin::uint>)
%shared_ptr(dolfin::VertexFunction<dolfin::uint>)

