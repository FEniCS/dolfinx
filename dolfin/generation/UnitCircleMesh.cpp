// Copyright (C) 2013 Garth N. Wells
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
// Modified by Garth N. Wells 2007
// Modified by Nuno Lopes 2008
// Modified by Anders Logg 2012
//
// First added:  2005-12-02
// Last changed: 2012-03-06

#include <dolfin/geometry/Point.h>
#include <dolfin/log/log.h>
#include "UnitCircleMesh.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
UnitCircleMesh::UnitCircleMesh(std::size_t n, std::string diagonal,
                                 std::string transformation) 
  : CircleMesh(Point(0.0, 0.0), 1.0, 1.0/n)
{
  deprecation("UnitCircleMesh",
              "1.3",
              "UnitCircleMesh is deprecated. Calling CircleMesh to create unstructured mesh instead");
}
//-----------------------------------------------------------------------------
