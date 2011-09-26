// Copyright (C) 2007 Kristian B. Oelgaard
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
// Modified by N. Lopes, 2008.
//
// First added:  2007-11-23
// Last changed: 2008-11-13

#include "dolfin/common/constants.h"
#include "CellType.h"
#include "MeshEditor.h"
#include "Interval.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
Interval::Interval(uint nx, double a, double b) : Mesh()
{
  if ( std::abs(a - b) < DOLFIN_EPS )
    error("Length of interval must be greater than zero.");

  if ( b < a )
    error("Length of interval is negative. Check the order of your arguments.");

  if ( nx < 1 )
    error("Number of points on interval must be at least 1.");

  rename("mesh", "Mesh of the interval (a, b)");

  // Open mesh for editing
  MeshEditor editor;
  editor.open(*this, CellType::interval, 1, 1);

  // Create vertices and cells:
  editor.init_vertices((nx+1));
  editor.init_cells(nx);

  // Create main vertices:
  for (uint ix = 0; ix <= nx; ix++)
  {
    const double x = a + (static_cast<double>(ix)*(b-a) / static_cast<double>(nx));
    editor.add_vertex(ix, x);
  }

  // Create intervals
  for (uint ix = 0; ix < nx; ix++)
  {
    const uint v0 = ix;
    const uint v1 = v0 + 1;
    editor.add_cell(ix, v0, v1);
  }

  // Close mesh editor
  editor.close();
}
//-----------------------------------------------------------------------------
