// Copyright (C) 2002 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#include <dolfin/Cell.h>
#include <dolfin/Triangle.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
int Triangle::noNodes()
{
  return 3;
}
//-----------------------------------------------------------------------------
int Triangle::noEdges()
{
  return 3;
}
//-----------------------------------------------------------------------------
int Triangle::noFaces()
{
  return 1;
}
//-----------------------------------------------------------------------------
int Triangle::noBoundaries()
{
  return noEdges();
}
//-----------------------------------------------------------------------------
Cell::Type Triangle::type()
{
  return Cell::TRIANGLE;
}
//-----------------------------------------------------------------------------
/*
//-----------------------------------------------------------------------------
real Triangle::ComputeVolume(Grid *grid)
{
  // The volume of a triangle is the same as its area
  
  // Get the coordinates
  Point *A = nodes[0]->GetCoord();
  Point *B = nodes[1]->GetCoord();
  Point *C = nodes[2]->GetCoord();

  // Make sure we get full precision
  real x1, x2, x3, y1, y2, y3, z1, z2, z3;

  x1 = real(A->x); y1 = real(A->y); z1 = real(A->z);
  x2 = real(B->x); y2 = real(B->y); z2 = real(B->z);
  x3 = real(C->x); y3 = real(C->y); z3 = real(C->z);

  // Formula for volume from http://mathworld.wolfram.com
  real v1 = (y1*z2 + z1*y3 + y2*z3) - ( y3*z2 + z3*y1 + y2*z1 );
  real v2 = (z1*x2 + x1*z3 + z2*x3) - ( z3*x2 + x3*z1 + z2*x1 );
  real v3 = (x1*y2 + y1*x3 + x2*y3) - ( x3*y2 + y3*x1 + x2*y1 );
  
  real v = 0.5 * sqrt( v1*v1 + v2*v2 + v3*v3 );

  return ( v );
}
//-----------------------------------------------------------------------------
real Triangle::ComputeCircumRadius(Grid *grid)
{
  // Compute volume
  real volume = ComputeVolume(grid);

  // Return radius
  return ( ComputeCircumRadius(grid,volume) );
}
//-----------------------------------------------------------------------------
real Triangle::ComputeCircumRadius(Grid *grid, real volume)
{
  // Get the coordinates
  Point *A = nodes[0]->GetCoord();
  Point *B = nodes[1]->GetCoord();
  Point *C = nodes[2]->GetCoord();
  
  // Compute side lengths
  real a  = B->Distance(*C);
  real b  = A->Distance(*C);
  real c  = A->Distance(*B);

  // Formula for radius from http://mathworld.wolfram.com
  real R = 0.25*a*b*c/volume;

  return ( R );
}
//-----------------------------------------------------------------------------
*/
//-----------------------------------------------------------------------------
bool Triangle::neighbor(ShortList<Node *> &cn, Cell &cell)
{
  // Two triangles are neighbors if they have a common edge or if they are
  // the same triangle, i.e. if they have 2 or 3 common nodes.

  if ( cell.type() != Cell::TRIANGLE )
	 return false;
  
  if ( !cell.c )
	 return false;

  int count = 0;
  for (int i = 0; i < 3; i++)
	 for (int j = 0; j < 3; j++)
		if ( cn(i) == cell.cn(j) )
		  count++;
  
  return count == 2 || count == 3;
}
//-----------------------------------------------------------------------------
