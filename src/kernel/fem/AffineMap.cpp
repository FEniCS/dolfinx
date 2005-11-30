// Copyright (C) 2005 Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// First added:  2005-05-17
// Last changed: 2005-11-29

#include <dolfin/Cell.h>
#include <dolfin/AffineMap.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
AffineMap::AffineMap()
{
  det = 0.0;
  
  f00 = 0.0; f01 = 0.0; f02 = 0.0;
  f10 = 0.0; f11 = 0.0; f12 = 0.0;
  f20 = 0.0; f21 = 0.0; f22 = 0.0;

  g00 = 0.0; g01 = 0.0; g02 = 0.0;
  g10 = 0.0; g11 = 0.0; g12 = 0.0;
  g20 = 0.0; g21 = 0.0; g22 = 0.0;

  _cell = 0;
}
//-----------------------------------------------------------------------------
AffineMap::~AffineMap()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
void AffineMap::update(Cell& cell)
{
  switch ( cell.type() )
  {
  case Cell::triangle:
    updateTriangle(cell);
    break;
  case Cell::tetrahedron:
    updateTetrahedron(cell);
    break;
  default:
    dolfin_error("Unknown cell type for affine map.");
  }
  
  _cell = &cell;
}
//-----------------------------------------------------------------------------
Point AffineMap::operator() (real X, real Y) const
{
  const real w = 1.0 - X - Y;
  return Point(w*p0.x + X*p1.x + Y*p2.x,
	       w*p0.y + X*p1.y + Y*p2.y);
}
//-----------------------------------------------------------------------------
Point AffineMap::operator() (real X, real Y, real Z) const
{
  const real w = 1.0 - X - Y - Z;
  return Point(w*p0.x + X*p1.x + Y*p2.x + Z*p3.x,
	       w*p0.y + X*p1.y + Y*p2.y + Z*p3.y,
	       w*p0.z + X*p1.z + Y*p2.z + Z*p3.z);
}
//-----------------------------------------------------------------------------
void AffineMap::updateTriangle(Cell& cell)
{
  dolfin_assert(cell.type() == Cell::triangle);
  
  // Get coordinates
  p0 = cell.coord(0);
  p1 = cell.coord(1);
  p2 = cell.coord(2);

  // Compute Jacobian of map
  f00 = p1.x - p0.x; f01 = p2.x - p0.x;
  f10 = p1.y - p0.y; f11 = p2.y - p0.y;
  
  // Compute determinant
  det = f00 * f11 - f01 * f10;
  
  // Check determinant
  if ( fabs(det) < DOLFIN_EPS )
    dolfin_error("Map from reference cell is singular.");
  
  // Compute inverse of Jacobian
  g00 =  f11 / det; g01 = -f01 / det;
  g10 = -f10 / det; g11 =  f00 / det;

  // Reset unused variables
  f02 = f12 = f20 = f21 = f22 = 0.0;
  g02 = g12 = g20 = g21 = g22 = 0.0;

  // Take absolute value of determinant
  det = fabs(det);
}
//-----------------------------------------------------------------------------
void AffineMap::updateTetrahedron(Cell& cell)
{
  dolfin_assert(cell.type() == Cell::tetrahedron);
  
  // Get coordinates
  p0 = cell.coord(0);
  p1 = cell.coord(1);
  p2 = cell.coord(2);
  p3 = cell.coord(3);
  
  // Compute Jacobian of map
  f00 = p1.x - p0.x; f01 = p2.x - p0.x; f02 = p3.x - p0.x;
  f10 = p1.y - p0.y; f11 = p2.y - p0.y; f12 = p3.y - p0.y;
  f20 = p1.z - p0.z; f21 = p2.z - p0.z; f22 = p3.z - p0.z;
  
  // Compute sub-determinants
  real d00 = f11*f22 - f12*f21;
  real d01 = f12*f20 - f10*f22;
  real d02 = f10*f21 - f11*f20;
  
  real d10 = f02*f21 - f01*f22;
  real d11 = f00*f22 - f02*f20;
  real d12 = f01*f20 - f00*f21;
  
  real d20 = f01*f12 - f02*f11;
  real d21 = f02*f10 - f00*f12;
  real d22 = f00*f11 - f01*f10;
  
  // Compute determinant
  det = f00 * d00 + f10 * d10 + f20 * d20;
  
  // Check determinant
  if ( fabs(det) < DOLFIN_EPS )
    dolfin_error("Map from reference cell is singular.");
  
  // Compute inverse of Jacobian
  g00 = d00 / det; g01 = d10 / det; g02 = d20 / det;
  g10 = d01 / det; g11 = d11 / det; g12 = d21 / det;
  g20 = d02 / det; g21 = d12 / det; g22 = d22 / det;

  // Take absolute value of determinant
  det = fabs(det);
}
//-----------------------------------------------------------------------------
