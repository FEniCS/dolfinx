// Copyright (C) 2006 Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// First added:  2006-06-05
// Last changed: 2006-10-19

#include <dolfin/dolfin_log.h>
#include <dolfin/Cell.h>
#include <dolfin/MeshEditor.h>
#include <dolfin/Triangle.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
dolfin::uint Triangle::dim() const
{
  return 2;
}
//-----------------------------------------------------------------------------
dolfin::uint Triangle::numEntities(uint dim) const
{
  switch ( dim )
  {
  case 0:
    return 3; // vertices
  case 1:
    return 3; // edges
  case 2:
    return 1; // cells
  default:
    dolfin_error1("Illegal topological dimension %d for tetrahedron.", dim);
  }

  return 0;
}
//-----------------------------------------------------------------------------
dolfin::uint Triangle::numVertices(uint dim) const
{
  switch ( dim )
  {
  case 0:
    return 1; // vertices
  case 1:
    return 2; // edges
  case 2:
    return 3; // cells
  default:
    dolfin_error1("Illegal topological dimension %d for tetrahedron.", dim);
  }

  return 0;
}
//-----------------------------------------------------------------------------
dolfin::uint Triangle::alignment(const Cell& cell, uint dim, uint e) const
{
  // Compute alignment according the convention in the DOLFIN manual
  if ( dim == 1 )
  {
    // Compute alignment of given edge by checking first vertex
    const uint* edge_vertices = cell.mesh().topology()(dim, 0)(cell.entities(dim)[e]);
    const uint* cell_vertices = cell.entities(0);
    return ( edge_vertices[0] == cell_vertices[(e + 1) % 3] ? 0 : 1 );
  }
  else
    dolfin_error("Unable to compute alignment for entity of dimension %d for triangle.");

  return 0;
}
//-----------------------------------------------------------------------------
void Triangle::createEntities(uint** e, uint dim, const uint v[]) const
{
  // We only need to know how to create edges
  if ( dim != 1 )
    dolfin_error1("Don't know how to create entities of topological dimension %d.", dim);

  // Create the three edges
  e[0][0] = v[1]; e[0][1] = v[2];
  e[1][0] = v[2]; e[1][1] = v[0];
  e[2][0] = v[0]; e[2][1] = v[1];
}
//-----------------------------------------------------------------------------
void Triangle::refineCell(Cell& cell, MeshEditor& editor,
			     uint& current_cell) const
{
  // Get vertices and edges
  const uint* v = cell.entities(0);
  const uint* e = cell.entities(1);
  dolfin_assert(v);
  dolfin_assert(e);

  // Get offset for new vertex indices
  const uint offset = cell.mesh().numVertices();

  // Compute indices for the six new vertices
  const uint v0 = v[0];
  const uint v1 = v[1];
  const uint v2 = v[2];
  const uint e0 = offset + e[0];
  const uint e1 = offset + e[1];
  const uint e2 = offset + e[2];
  
  // Add the four new cells
  editor.addCell(current_cell++, v0, e2, e1);
  editor.addCell(current_cell++, v1, e0, e2);
  editor.addCell(current_cell++, v2, e1, e0);
  editor.addCell(current_cell++, e0, e1, e2);
}
//-----------------------------------------------------------------------------
std::string Triangle::description() const
{
  std::string s = "triangle (simplex of topological dimension 2)";
  return s;
}
//-----------------------------------------------------------------------------
