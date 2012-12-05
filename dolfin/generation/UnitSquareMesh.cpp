// Copyright (C) 2005-2009 Anders Logg
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
// Modified by Garth N. Wells 2007.
// Modified by Kristian B. Oelgaard 2009.
//
// First added:  2005-12-02
// Last changed: 2009-09-29

#include "UnitSquareMesh.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
UnitSquareMesh::UnitSquareMesh(std::size_t nx, std::size_t ny,
   std::string diagonal) : RectangleMesh(0.0, 0.0, 1.0, 1.0, nx, ny, diagonal)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
