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
Point P1TriMap::operator() (const Point& p) const
{
  // Check that we have a cell
  if ( !_cell )
    dolfin_error("Unable to evaluate map to cell. No cell given.");

  // Check that cell type is correct
  if ( _cell->type() != Cell::triangle )
    dolfin_error("Wrong cell type for map (must be a triangle).");

  // Get the points of the cell
  Point& p0 = _cell->coord(0);
  Point& p1 = _cell->coord(1);
  Point& p2 = _cell->coord(2);

  // Evaluate basis functions
  real phi0 = 1.0 - p.x - p.y;
  real phi1 = p.x;
  real phi2 = p.y;

  // Map point
  Point q;
  q.x = phi0*p0.x + phi1*p1.x + phi2*p2.x;
  q.y = phi0*p0.y + phi1*p1.y + phi2*p2.y;
  q.z = phi0*p0.z + phi1*p1.z + phi2*p2.z;

  return q;
}
//-----------------------------------------------------------------------------
Point P1TriMap::operator() (const Point& p, unsigned int boundary) const
{
  // Check that we have a cell
  if ( !_cell )
    dolfin_error("Unable to evaluate map to cell. No cell given.");

  // Check that cell type is correct
  if ( _cell->type() != Cell::triangle )
    dolfin_error("Wrong cell type for map (must be a triangle).");

  // Get the points of the cell
  Point& p0 = _cell->coord(0);
  Point& p1 = _cell->coord(1);
  Point& p2 = _cell->coord(2);

  // Map point
  Point q;
  switch (boundary) {
  case 0: // Map to edge between n0 and n1
    q.x = p0.x + p.x*(p1.x - p0.x);
    q.y = p0.y + p.x*(p1.y - p0.y);
    q.z = p0.z + p.x*(p1.z - p0.z);
    break;
  case 1: // Map to edge between n1 and n2
    q.x = p1.x + p.x*(p2.x - p1.x);
    q.y = p1.y + p.x*(p2.y - p1.y);
    q.z = p1.z + p.x*(p2.z - p1.z);
    break;
  case 2: // Map to edge between n2 and n0
    q.x = p2.x + p.x*(p0.x - p2.x);
    q.y = p2.y + p.x*(p0.y - p2.y);
    q.z = p2.z + p.x*(p0.z - p2.z);
    break;
  default:
    dolfin_error("Illegal boundary number for triangle, must be 0, 1, or 2.");
  }

  return q;
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
