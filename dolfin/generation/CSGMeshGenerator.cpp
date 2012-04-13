// Copyright (C) 2012 Anders Logg (and others, add authors)
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
// First added:  2012-01-01
// Last changed: 2012-04-13

#include <dolfin/log/log.h>
#include "CSGMeshGenerator.h"
#include "CSGGeometry.h"

// FIXME: Temporary includes
#include "UnitSquare.h"
#include "UnitCube.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
void CSGMeshGenerator::generate(Mesh& mesh,
                                const CSGGeometry& geometry)
{
  info("Generating mesh from CSG... not implemented");

  // Put CGAL implementation here and in private static functions

  // Temporary implementation just to generate something
  if (geometry.dim() == 2)
  {
    UnitSquare unit_square(8, 8);
    mesh = unit_square;
  }
  else if (geometry.dim() == 3)
  {
    UnitCube unit_cube(8, 8, 8);
    mesh = unit_cube;
  }
  else
  {
    dolfin_error("CSGMeshGenerator.cpp",
                 "create mesh from CSG geometry",
                 "Unhandled geometry dimension %d", geometry.dim());
  }
}
//-----------------------------------------------------------------------------
