#include <utils.h>

#include <Display.hh>
#include "Edge.hh"
#include "Node.hh"
#include "Grid.hh"

//-----------------------------------------------------------------------------
void Edge::Set(int en1, int en2, int mn)
{
  end_nodes[0] = en1;
  end_nodes[1] = en2;

  mid_node = mn;
}
//-----------------------------------------------------------------------------
void Edge::SetEndnodes(int en1, int en2)
{
  end_nodes[0] = en1;
  end_nodes[1] = en2;
}
//-----------------------------------------------------------------------------
void Edge::SetMidnode(int mn)
{
  mid_node = mn;
}
//-----------------------------------------------------------------------------
int Edge::GetEndnode(int node)
{
  if ( (node<0) || (node>=2) )
	 display->InternalError("Edge::GetEndNode()","Illegal node: %d",node);
  
  return end_nodes[node];
}
//-----------------------------------------------------------------------------
int Edge::GetMidnode()
{
  return mid_node;
}
//-----------------------------------------------------------------------------
real Edge::ComputeLength(Grid *grid)
{
  // Get the coordinates
  Point A = grid->GetNode(end_nodes[0])->GetCoord();
  Point B = grid->GetNode(end_nodes[1])->GetCoord();

  // Make sure we get full precision
  real x1, x2, y1, y2, z1, z2;

  x1 = real(A.x); y1 = real(A.y); z1 = real(A.z);
  x2 = real(B.x); y2 = real(B.y); z2 = real(B.z);

  // The Euklidian length of the edge 
  real l = sqrt( sqr(x2-x1) + sqr(y2-y1) + sqr(z2-z1) );
  
  return ( l );
}
//-----------------------------------------------------------------------------
Point Edge::ComputeMidpoint(Grid *grid)
{
  // Get the coordinates
  Point A = grid->GetNode(end_nodes[0])->GetCoord();
  Point B = grid->GetNode(end_nodes[1])->GetCoord();

  // Make sure we get full precision
  real x1, x2, y1, y2, z1, z2;

  x1 = real(A.x); y1 = real(A.y); z1 = real(A.z);
  x2 = real(B.x); y2 = real(B.y); z2 = real(B.z);

  // The midpoint of the edge 
  Point M;
  M.x = 0.5*(x1+x2);
  M.y = 0.5*(y1+y2);
  M.z = 0.5*(z1+z2);
  
  return ( M );
}
//-----------------------------------------------------------------------------
