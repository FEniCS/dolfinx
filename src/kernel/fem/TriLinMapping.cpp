// Copyright (C) 2003 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#include <dolfin/Cell.h>
#include <dolfin/Point.h>
#include <dolfin/Node.h>
#include <dolfin/TriLinMapping.h>

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
  return g11*v.dx() + g21*v.dy();
}
//-----------------------------------------------------------------------------
const FunctionSpace::ElementFunction TriLinMapping::dy
(const FunctionSpace::ShapeFunction &v) const
{
  return g12*v.dx() + g22*v.dy();
}
//-----------------------------------------------------------------------------
const FunctionSpace::ElementFunction TriLinMapping::dz
(const FunctionSpace::ShapeFunction &v) const
{
  return v.dz();
}
//-----------------------------------------------------------------------------
const FunctionSpace::ElementFunction TriLinMapping::dt
(const FunctionSpace::ShapeFunction &v) const
{
  return v.dt();
}
//-----------------------------------------------------------------------------
void TriLinMapping::update(const Cell& cell)
{
  // Check that cell type is correct
  if ( cell.type() != Cell::TRIANGLE ) {
	 // FIXME: Use logging system
	 cout << "Error: Wrong cell type for mapping (must be a triangle)." << endl;
	 exit(1);
  }
  
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
  if ( fabs(d) < DOLFIN_EPS ) {
	 // FIXME: Use logging system
	 cout << "Error: mapping from reference element is singular." << endl;
	 exit(1);
  }
  
  // Compute inverse
  g11 =   f22 / d; g12 = - f12 / d;
  g21 = - f21 / d; g22 =   f11 / d;
}
//-----------------------------------------------------------------------------
