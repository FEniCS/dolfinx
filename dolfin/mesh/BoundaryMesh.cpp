// Copyright (C) 2006-2012 Anders Logg
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
// Modified by Niclas Jansson 2009.
// Modified by Joachim B Haga 2012.
//
// First added:  2006-06-21
// Last changed: 2012-09-05

#include <iostream>

#include <dolfin/log/log.h>
#include "BoundaryComputation.h"
#include "BoundaryMesh.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
BoundaryMesh::BoundaryMesh() : Mesh()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
BoundaryMesh::BoundaryMesh(const Mesh& mesh, bool order) : Mesh()
{
  // Create boundary mesh
  init_exterior_boundary(mesh);

  // Order mesh if requested
  if (order)
    this->order();
}
//-----------------------------------------------------------------------------
BoundaryMesh::~BoundaryMesh()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
void BoundaryMesh::init_exterior_boundary(const Mesh& mesh)
{
  BoundaryComputation::compute_exterior_boundary(mesh, *this);
}
//-----------------------------------------------------------------------------
void BoundaryMesh::init_interior_boundary(const Mesh& mesh)
{
  BoundaryComputation::compute_interior_boundary(mesh, *this);
}
//-----------------------------------------------------------------------------
const MeshFunction<std::size_t>& BoundaryMesh::entity_map(std::size_t d) const
{ 
  if (d == 0)
    return _vertex_map; 
  else if (d == this->topology().dim())
    return _cell_map; 
  else
  {
    dolfin_error("BoundaryMesh.cpp",
                 "access entity map (from boundary mesh underlying mesh",
                 "Can onlt access vertex and cells maps");
  }

  // Return something to keep compilers happy, code is never reached
  return _cell_map;
}
//-----------------------------------------------------------------------------

