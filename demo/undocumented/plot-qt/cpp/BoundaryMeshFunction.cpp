// Copyright (C) 2012 Joachim Berdal Haga
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
// First added:  2012-09-14
// Last changed: 2012-09-18

#include "BoundaryMeshFunction.h"

using namespace dolfin;

//----------------------------------------------------------------------------
BoundaryMeshFunction::BoundaryMeshFunction(const Mesh& mesh)
  : _bmesh(mesh)
{
  MeshFunction<bool>::init(_bmesh, _bmesh.topology().dim());
  set_all(false);
}
//----------------------------------------------------------------------------
void BoundaryMeshFunction::toggleCell(int id)
{
  (*this)[id] = !(*this)[id];
}
//----------------------------------------------------------------------------
