// Copyright (C) 2003 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#include <dolfin/dolfin_log.h>
#include <dolfin/Cell.h>
#include <dolfin/Point.h>
#include <dolfin/Node.h>
#include <dolfin/TriLinMapping.h>
#include <cmath>

using namespace dolfin;

//-----------------------------------------------------------------------------
TriLinMapping::TriLinMapping() : Mapping()
{
  // Set dimension
  dim = 2;
}
//-----------------------------------------------------------------------------
const FunctionSpace::ElementFunction TriLinMapping::dx
(const FunctionSpace::ShapeFunction &v) const
{
  return g11*v.dX() + g21*v.dY();
}
//-----------------------------------------------------------------------------
const FunctionSpace::ElementFunction TriLinMapping::dy
(const FunctionSpace::ShapeFunction &v) const
{
  return g12*v.dX() + g22*v.dY();
}
//-----------------------------------------------------------------------------
const FunctionSpace::ElementFunction TriLinMapping::dz
(const FunctionSpace::ShapeFunction &v) const
{
  return v.dZ();
}
//-----------------------------------------------------------------------------
const FunctionSpace::ElementFunction TriLinMapping::dt
(const FunctionSpace::ShapeFunction &v) const
{
  return v.dT();
}
//-----------------------------------------------------------------------------
void TriLinMapping::update(const Cell& cell)
{
  // Check that cell type is correct
  if ( cell.type() != Cell::TRIANGLE )
	 dolfin_error("Wrong cell type for mapping (must be a triangle).");
  
  // Reset values
  reset();

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
	 dolfin_error("Mapping from reference element is singular.");
  
  // Compute inverse
  g11 =   f22 / d; g12 = - f12 / d;
  g21 = - f21 / d; g22 =   f11 / d;
}
//-----------------------------------------------------------------------------
