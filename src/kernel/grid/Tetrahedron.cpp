// Copyright (C) 2002 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#include <dolfin/Cell.h>
#include <dolfin/Tetrahedron.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
int Tetrahedron::noNodes()
{
  return 4;
}
//-----------------------------------------------------------------------------
int Tetrahedron::noEdges()
{
  return 6;
}
//-----------------------------------------------------------------------------
int Tetrahedron::noFaces()
{
  return 4;
}
//-----------------------------------------------------------------------------
int Tetrahedron::noBoundaries()
{
  return noFaces();
}
//-----------------------------------------------------------------------------
Cell::Type Tetrahedron::type()
{
  return Cell::TETRAHEDRON;
}
//-----------------------------------------------------------------------------
/*
real Tetrahedron::ComputeVolume(Grid *grid)
{
  // Get the coordinates
  Point *A = nodes[0]->GetCoord();
  Point *B = nodes[1]->GetCoord();
  Point *C = nodes[2]->GetCoord();
  Point *D = nodes[3]->GetCoord();

  // Make sure we get full precision
  real x1, x2, x3, x4, y1, y2, y3, y4, z1, z2, z3, z4;

  x1 = real(A->x); y1 = real(A->y); z1 = real(A->z);
  x2 = real(B->x); y2 = real(B->y); z2 = real(B->z);
  x3 = real(C->x); y3 = real(C->y); z3 = real(C->z);
  x4 = real(D->x); y4 = real(D->y); z4 = real(D->z);

  // Formula for volume from http://mathworld.wolfram.com
  real v = ( x1 * ( y2*z3 + y4*z2 + y3*z4 - y3*z2 - y2*z4 - y4*z3 ) -
				 x2 * ( y1*z3 + y4*z1 + y3*z4 - y3*z1 - y1*z4 - y4*z3 ) +
				 x3 * ( y1*z2 + y4*z1 + y2*z4 - y2*z1 - y1*z4 - y4*z2 ) -
				 x4 * ( y1*z2 + y2*z3 + y3*z1 - y2*z1 - y3*z2 - y1*z3 ) );
  
  //display->Message(0,"A = (%f,%f,%f)\nB = (%f,%f,%f)\nC = (%f,%f,%f)\nD = (%f,%f,%f)\n",
  //						 x1,y1,z1,
  //						 x2,y2,z2,
  //						 x3,y3,z3,
  //						 x4,y4,z4);
  
  v = ( v >= 0.0 ? v : -v );
  
  return ( v );
}
//-----------------------------------------------------------------------------
real Tetrahedron::ComputeCircumRadius(Grid *grid)
{
  // Compute volume
  real volume = ComputeVolume(grid);

  // Compute radius
  real radius = ComputeCircumRadius(grid,volume);

  //display->Message(0,"v = %f R = %f",volume,radius);
  
  // Return radius
  return ( radius );
}
//-----------------------------------------------------------------------------
real Tetrahedron::ComputeCircumRadius(Grid *grid, real volume)
{
  // Get the coordinates
  Point *A = nodes[0]->GetCoord();
  Point *B = nodes[1]->GetCoord();
  Point *C = nodes[2]->GetCoord();
  Point *D = nodes[3]->GetCoord();
  
  // Compute side lengths
  real a  = B->Distance(*C);
  real b  = A->Distance(*C);
  real c  = A->Distance(*B);
  real aa = A->Distance(*D);
  real bb = B->Distance(*D);
  real cc = C->Distance(*D);
  
  // Compute "area" of triangle with strange side lengths
  real l1   = a*aa;
  real l2   = b*bb;
  real l3   = c*cc;
  real s    = 0.5*(l1+l2+l3);
  real area = sqrt(s*(s-l1)*(s-l2)*(s-l3));

  // Formula for radius from http://mathworld.wolfram.com
  real R = area/(6.0*volume);

  return ( R );
}
*/
//-----------------------------------------------------------------------------
bool Tetrahedron::neighbor(ShortList<Node *> &cn, Cell &cell)
{
  // Two tetrahedrons are neighbors if they have a common face or if they are
  // the same tetrahedron, i.e. if they have 3 or 4 common nodes.
  
  if ( cell.type() != Cell::TETRAHEDRON )
	 return false;
  
  if ( !cell.c )
	 return false;

  int count = 0;
  for (int i = 0; i < 4; i++)
	 for (int j = 0; j < 4; j++)
		if ( cn(i) == cell.cn(j) )
		  count++;
  
  return count == 3 || count == 4;
}
//-----------------------------------------------------------------------------
