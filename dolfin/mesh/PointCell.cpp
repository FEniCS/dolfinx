// Copyright (C) 2007-2007 Kristian B. Oelgaard.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2007-12-12
// Last changed: 2007-12-12
//

#include <dolfin/log/dolfin_log.h>
#include "Cell.h"
#include "MeshEditor.h"
#include "Facet.h"
#include "PointCell.h"
#include "Vertex.h"
#include "GeometricPredicates.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
dolfin::uint PointCell::dim() const
{
  return 0;
}
//-----------------------------------------------------------------------------
dolfin::uint PointCell::numEntities(uint dim) const
{
  switch ( dim )
  {
  case 0:
    return 1; // vertices
  default:
    error("Illegal topological dimension %d for point.", dim);
  }

  return 0;
}
//-----------------------------------------------------------------------------
dolfin::uint PointCell::numVertices(uint dim) const
{
  switch ( dim )
  {
  case 0:
    return 1; // vertices
  default:
    error("Illegal topological dimension %d for point.", dim);
  }

  return 0;
}
//-----------------------------------------------------------------------------
dolfin::uint PointCell::orientation(const Cell& cell) const
{
    error("PointCell::orientation() not defined.");
  return 0;
}
//-----------------------------------------------------------------------------
void PointCell::createEntities(uint** e, uint dim, const uint* v) const
{
    error("PointCell::createEntities() don't know how to create entities on a point.");
}
//-----------------------------------------------------------------------------
void PointCell::orderEntities(Cell& cell) const
{
    error("PointCell::orderEntities() not defined.");
}
//-----------------------------------------------------------------------------
void PointCell::refineCell(Cell& cell, MeshEditor& editor,
                          uint& current_cell) const
{
    error("PointCell::refineCell() not defined.");
}
//-----------------------------------------------------------------------------
real PointCell::volume(const MeshEntity& triangle) const
{
    error("PointCell::volume() not defined.");
  return 0.0;
}
//-----------------------------------------------------------------------------
real PointCell::diameter(const MeshEntity& triangle) const
{
    error("PointCell::diameter() not defined.");
  return 0.0;
}
//-----------------------------------------------------------------------------
real PointCell::normal(const Cell& cell, uint facet, uint i) const
{
    error("PointCell::normal() not defined.");
  return 0.0;
}
//-----------------------------------------------------------------------------
bool PointCell::intersects(const MeshEntity& triangle, const Point& p) const
{
    error("PointCell::intersects() not implemented.");
  return true;
}
//-----------------------------------------------------------------------------
bool PointCell::intersects(const MeshEntity& triangle, const Point& p1, const Point& p2) const
{
    error("PointCell::intersects() not implemented.");
  return true;
}
//-----------------------------------------------------------------------------
std::string PointCell::description() const
{
  std::string s = "point (simplex of topological dimension 0)";
  return s;
}
//-----------------------------------------------------------------------------
dolfin::uint PointCell::findEdge(uint i, const Cell& cell) const
{
    error("PointCell::findEdge() not defined.");
  return 0;
}
//-----------------------------------------------------------------------------
