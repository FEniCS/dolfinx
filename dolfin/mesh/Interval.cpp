// Copyright (C) 2007 Kristian B. Oelgaard.
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by N. Lopes, 2008.
//
// First added:  2007-11-23
// Last changed: 2008-11-113

#include "MeshEditor.h"
#include "Interval.h"
#include "dolfin/common/constants.h"

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
