// Copyright (C) 2003 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#include <dolfin/dolfin_log.h>
#include <dolfin/Cell.h>
#include <dolfin/Point.h>
#include <dolfin/Node.h>
#include <dolfin/TetLinMapping.h>
#include <cmath>

using namespace dolfin;

//-----------------------------------------------------------------------------
TetLinMapping::TetLinMapping() : Mapping()
{
  // Set dimension
  dim = 3;
}
//-----------------------------------------------------------------------------
const FunctionSpace::ElementFunction TetLinMapping::dx
(const FunctionSpace::ShapeFunction &v) const
{
  return g11*v.dX() + g21*v.dY() + g31*v.dZ();
}
//-----------------------------------------------------------------------------
const FunctionSpace::ElementFunction TetLinMapping::dy
(const FunctionSpace::ShapeFunction &v) const
{
  return g12*v.dX() + g22*v.dY() + g32*v.dZ();
}
//-----------------------------------------------------------------------------
const FunctionSpace::ElementFunction TetLinMapping::dz
(const FunctionSpace::ShapeFunction &v) const
{
  return g13*v.dX() + g23*v.dY() + g33*v.dZ();
}
//-----------------------------------------------------------------------------
const FunctionSpace::ElementFunction TetLinMapping::dt
(const FunctionSpace::ShapeFunction &v) const
{
  return v.dT();
}
//-----------------------------------------------------------------------------
void TetLinMapping::update(const Cell& cell)
{
  // Check that cell type is correct
  if ( cell.type() != Cell::TETRAHEDRON )
	 dolfin_error("Wrong cell type for mapping (must be a tetrahedron).");
  
  // Reset values
  reset();

  // Get coordinates
  NodeIterator n(cell);
  Point p0 = n->coord(); ++n;
  Point p1 = n->coord(); ++n;
  Point p2 = n->coord(); ++n;
  Point p3 = n->coord();

  // Set values for Jacobian
  f11 = p1.x - p0.x; f12 = p2.x - p0.x; f13 = p3.x - p0.x;
  f21 = p1.y - p0.y; f22 = p2.y - p0.y; f23 = p3.y - p0.y;
  f31 = p1.z - p0.z; f32 = p2.z - p0.z; f33 = p3.z - p0.z;

  // Compute sub-determinants
  real d11 = f22*f33 - f23*f32;
  real d12 = f23*f31 - f21*f33;
  real d13 = f21*f32 - f22*f31;
  
  real d21 = f13*f32 - f12*f33;
  real d22 = f11*f33 - f13*f31;
  real d23 = f12*f31 - f11*f32;
  
  real d31 = f12*f23 - f13*f22;
  real d32 = f13*f21 - f11*f23;
  real d33 = f11*f22 - f12*f21;
  
  // Compute determinant
  d = f11 * d11 + f21 * d21 + f31 * d31;

  // Check determinant
  if ( fabs(d) < DOLFIN_EPS )
	 dolfin_error("Mapping from reference element is singular.");
  
  // Compute inverse
  g11 = d11 / d; g12 = d21 / d; g13 = d31 / d;
  g21 = d12 / d; g22 = d22 / d; g23 = d32 / d;
  g31 = d13 / d; g32 = d23 / d; g33 = d33 / d;
}
//-----------------------------------------------------------------------------
