// Copyright (C) 2003 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#include <dolfin/dolfin_log.h>
#include <dolfin/Cell.h>
#include <dolfin/Face.h>
#include <dolfin/Edge.h>
#include <dolfin/Point.h>
#include <dolfin/Node.h>
#include <dolfin/P1TetMap.h>
#include <cmath>

using namespace dolfin;

//-----------------------------------------------------------------------------
P1TetMap::P1TetMap() : Map()
{
  // Set dimension
  dim = 3;
}
//-----------------------------------------------------------------------------
void P1TetMap::update(const Cell& cell)
{
  // Check that cell type is correct
  if ( cell.type() != Cell::tetrahedron )
    dolfin_error("Wrong cell type for map (must be a tetrahedron).");
  
  cell_ = &cell;

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
	 dolfin_error("Map from reference element is singular.");
  
  // Compute inverse
  g11 = d11 / d; g12 = d21 / d; g13 = d31 / d;
  g21 = d12 / d; g22 = d22 / d; g23 = d32 / d;
  g31 = d13 / d; g32 = d23 / d; g33 = d33 / d;
}
//-----------------------------------------------------------------------------
const FunctionSpace::ElementFunction P1TetMap::ddx
(const FunctionSpace::ShapeFunction &v) const
{
  return g11*v.ddX() + g21*v.ddY() + g31*v.ddZ();
}
//-----------------------------------------------------------------------------
const FunctionSpace::ElementFunction P1TetMap::ddy
(const FunctionSpace::ShapeFunction &v) const
{
  return g12*v.ddX() + g22*v.ddY() + g32*v.ddZ();
}
//-----------------------------------------------------------------------------
const FunctionSpace::ElementFunction P1TetMap::ddz
(const FunctionSpace::ShapeFunction &v) const
{
  return g13*v.ddX() + g23*v.ddY() + g33*v.ddZ();
}
//-----------------------------------------------------------------------------
const FunctionSpace::ElementFunction P1TetMap::ddt
(const FunctionSpace::ShapeFunction &v) const
{
  return v.ddT();
}
//-----------------------------------------------------------------------------
void P1TetMap::update(const Edge& bnd_edge)
{
  dolfin_error("Edge map not implemented for 3D maps.");
}
//-----------------------------------------------------------------------------
void P1TetMap::update(const Face& bnd_face)
{
  // Update map to interior of cell
  update(bnd_face.cell(0));

  // The determinant is given by the norm of the cross product
  
  // Get the first two edges
  Edge& e0 = bnd_face.edge(0);
  Edge& e1 = bnd_face.edge(1);

  // Get coordinates
  Point& p00 = e0.coord(0);
  Point& p01 = e0.coord(1);
  Point& p10 = e1.coord(0);
  Point& p11 = e1.coord(1);

  // Compute vectors
  real dx0 = p01.x - p00.x;
  real dy0 = p01.y - p00.y;
  real dz0 = p01.z - p00.z;
  real dx1 = p11.x - p10.x;
  real dy1 = p11.y - p10.y;
  real dz1 = p11.z - p10.z;

  // Compute cross-product
  real dx = dy0*dz1 - dz0*dy1;
  real dy = dz0*dx1 - dx0*dz1;
  real dz = dx0*dy1 - dy0*dx1;

  // Compute norm
  bd = sqrt(dx*dx + dy*dy + dz*dz);

  // Check determinant
  if ( fabs(bd) < DOLFIN_EPS )
    dolfin_error("Map to boundary of cell is singular.");
}
//-----------------------------------------------------------------------------
