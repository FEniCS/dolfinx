// Copyright (C) 2006 Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// First added:  2006-06-05
// Last changed: 2006-08-08

#include <dolfin/dolfin_log.h>
#include <dolfin/Cell.h>
#include <dolfin/MeshEditor.h>
#include <dolfin/Tetrahedron.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
dolfin::uint Tetrahedron::numEntities(uint dim) const
{
  switch ( dim )
  {
  case 0:
    return 4; // vertices
  case 1:
    return 6; // edges
  case 2:
    return 4; // faces
  case 3:
    return 1; // cells
  default:
    dolfin_error1("Illegal topological dimension %d for tetrahedron.", dim);
  }

  return 0;
}
//-----------------------------------------------------------------------------
dolfin::uint Tetrahedron::numVertices(uint dim) const
{
  switch ( dim )
  {
  case 0:
    return 1; // vertices
  case 1:
    return 2; // edges
  case 2:
    return 3; // faces
  case 3:
    return 4; // cells
  default:
    dolfin_error1("Illegal topological dimension %d for tetrahedron.", dim);
  }

  return 0;
}
//-----------------------------------------------------------------------------
void Tetrahedron::createEntities(uint** e, uint dim, const uint v[]) const
{
  // We only need to know how to create edges and faces
  switch ( dim )
  {
  case 1:
    // Create the six edges
    e[0][0] = v[1]; e[0][1] = v[2];
    e[1][0] = v[2]; e[1][1] = v[0];
    e[2][0] = v[0]; e[2][1] = v[1];
    e[3][0] = v[0]; e[3][1] = v[3];
    e[4][0] = v[1]; e[4][1] = v[3];
    e[5][0] = v[2]; e[5][1] = v[3];
    break;
  case 2:
    // Create the four faces
    e[0][0] = v[1]; e[0][1] = v[3]; e[0][2] = v[2];
    e[1][0] = v[2]; e[1][1] = v[3]; e[1][2] = v[0];
    e[2][0] = v[3]; e[2][1] = v[1]; e[2][2] = v[0];
    e[3][0] = v[0]; e[3][1] = v[1]; e[3][2] = v[2];
    break;
  default:
    dolfin_error1("Don't know how to create entities of topological dimension %d.", dim);
  }
}
//-----------------------------------------------------------------------------
void Tetrahedron::refineCell(Cell& cell, MeshEditor& editor,
				uint& current_cell) const
{
  // Get vertices and edges
  const uint* v = cell.connections(0);
  const uint* e = cell.connections(1);
  dolfin_assert(v);
  dolfin_assert(e);

  // Get offset for new vertex indices
  const uint offset = cell.mesh().numVertices();

  // Compute indices for the ten new vertices
  const uint v0 = v[0];
  const uint v1 = v[1];
  const uint v2 = v[2];
  const uint v3 = v[3];
  const uint e0 = offset + e[0];
  const uint e1 = offset + e[1];
  const uint e2 = offset + e[2];
  const uint e3 = offset + e[3];
  const uint e4 = offset + e[4];
  const uint e5 = offset + e[5];
  
  // Add the eight new cells
  editor.addCell(current_cell++, v0, e1, e3, e2);
  editor.addCell(current_cell++, v1, e2, e4, e0);
  editor.addCell(current_cell++, v2, e0, e5, e1);
  editor.addCell(current_cell++, v3, e5, e4, e3);
  editor.addCell(current_cell++, e0, e4, e5, e1);
  editor.addCell(current_cell++, e1, e5, e3, e4);
  editor.addCell(current_cell++, e2, e3, e4, e1);
  editor.addCell(current_cell++, e0, e1, e2, e4);
}
//-----------------------------------------------------------------------------
std::string Tetrahedron::description() const
{
  std::string s = "tetrahedron (simplex of topological dimension 3)";
  return s;
}
//-----------------------------------------------------------------------------
