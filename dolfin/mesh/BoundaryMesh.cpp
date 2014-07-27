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
BoundaryMesh::BoundaryMesh(const Mesh& mesh, std::string type, bool order)
  : Mesh()
{
  const std::size_t dim = mesh.topology().dim();
  if (mesh.topology().ghost_offset(dim) != mesh.topology().size(dim))
  {
    dolfin_error("BoundaryMesh.cpp",
                 "create BoundaryMesh with ghost cells",
                 "Disable ghost mesh");
  }

  // Create boundary mesh
  BoundaryComputation::compute_boundary(mesh, type, *this);

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
MeshFunction<std::size_t>& BoundaryMesh::entity_map(std::size_t d)
{
  if (d == 0)
    return _vertex_map;
  else if (d == this->topology().dim())
    return _cell_map;
  else
  {
    dolfin_error("BoundaryMesh.cpp",
                 "access entity map (from boundary mesh underlying mesh",
                 "Can only access vertex and cells maps");
  }

  // Return something to keep compilers happy, code is never reached
  return _cell_map;
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
                 "Can only access vertex and cells maps");
  }

  // Return something to keep compilers happy, code is never reached
  return _cell_map;
}
//-----------------------------------------------------------------------------
