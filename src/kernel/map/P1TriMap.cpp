// Copyright (C) 2003 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#include <dolfin/dolfin_log.h>
#include <dolfin/Cell.h>
#include <dolfin/Edge.h>
#include <dolfin/Point.h>
#include <dolfin/Node.h>
#include <dolfin/P1TriMap.h>
#include <cmath>

using namespace dolfin;

//-----------------------------------------------------------------------------
P1TriMap::P1TriMap() : Map()
{
  // Set dimension
  dim = 2;
}
//-----------------------------------------------------------------------------
void P1TriMap::update(const Cell& cell)
{
  // Check that cell type is correct
  if ( cell.type() != Cell::triangle )
    dolfin_error("Wrong cell type for map (must be a triangle).");

  // Reset values: note that this includes setting _boundary to -1,
  // denoting a mapping to the interior of the cell
  reset();

  // Update current cell
  _cell = &cell;
  
  // Get coordinates
  NodeIterator n(cell);
  Point p0 = n->coord(); ++n;
  Point p1 = n->coord(); ++n;
  Point p2 = n->coord();

  // Set values for Jacobian
  f11 = p1.x - p0.x; f12 = p2.x - p0.x;
  f21 = p1.y - p0.y; f22 = p2.y - p0.y;
  
  // Compute determinant
  d = f11 * f22 - f12 * f21;
  
  // Check determinant
  if ( fabs(d) < DOLFIN_EPS )
    dolfin_error("Map from reference element is singular.");
  
  // Compute inverse
  g11 =   f22 / d; g12 = - f12 / d;
  g21 = - f21 / d; g22 =   f11 / d;
}
//-----------------------------------------------------------------------------
void P1TriMap::update(const Edge& edge)
{
  // Check that there is only one cell neighbor
  if ( edge.noCellNeighbors() != 1 )
  {
    cout << "Updating map to edge on boundary: " << edge << endl;
    dolfin_error("Edge on boundary does not belong to exactly one cell.");
  }

  // Get the cell neighbor of the edge
  Cell& cell = edge.cell(0);

  // Update map to interior of cell
  update(cell);

  // Compute edge number
  _boundary = edgeNumber(edge, cell);

  // The determinant is given by the length of the edge
  bd = edge.length();

  // Check determinant
  if ( fabs(bd) < DOLFIN_EPS )
    dolfin_error("Map to boundary of cell is singular.");
}
//-----------------------------------------------------------------------------
const FunctionSpace::ElementFunction P1TriMap::ddx
(const FunctionSpace::ShapeFunction& v) const
{
  return g11*v.ddX() + g21*v.ddY();
}
//-----------------------------------------------------------------------------
const FunctionSpace::ElementFunction P1TriMap::ddy
(const FunctionSpace::ShapeFunction& v) const
{
  return g12*v.ddX() + g22*v.ddY();
}
//-----------------------------------------------------------------------------
const FunctionSpace::ElementFunction P1TriMap::ddz
(const FunctionSpace::ShapeFunction& v) const
{
  return v.ddZ();
}
//-----------------------------------------------------------------------------
const FunctionSpace::ElementFunction P1TriMap::ddt
(const FunctionSpace::ShapeFunction& v) const
{
  return v.ddT();
}
//-----------------------------------------------------------------------------
unsigned int P1TriMap::edgeNumber(const Edge& edge, const Cell& cell) const
{
  // The local ordering of faces within the cells should automatically take
  // care of this, but in the meantime we need to compute the number of
  // the given face by hand. See documentation for the Mesh class for
  // details on the ordering.

  // Not implemented

  return 0;
}
//-----------------------------------------------------------------------------
