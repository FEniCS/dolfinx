#include <utils.h>

#include <dolfin/Display.hh>
#include <dolfin/Tetrahedron.hh>
#include <dolfin/Node.hh>
#include <dolfin/Grid.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
Tetrahedron::Tetrahedron()
{
  nodes[0] = 0;
  nodes[1] = 0;
  nodes[2] = 0;
  nodes[3] = 0;
}
//-----------------------------------------------------------------------------
Tetrahedron::~Tetrahedron()
{

}
//-----------------------------------------------------------------------------
void Tetrahedron::set(Node *n0, Node *n1, Node *n2, Node *n3)
{
  nodes[0] = n0;
  nodes[1] = n1;
  nodes[2] = n2;
  nodes[3] = n3;
}
//-----------------------------------------------------------------------------
void Tetrahedron::Set(Node *n1, Node *n2, Node *n3, Node *n4, int material)
{
  nodes[0] = n1;
  nodes[1] = n2;
  nodes[2] = n3;
  nodes[3] = n4;
  
  this->material = material;

  id = -1;
}
//-----------------------------------------------------------------------------
int Tetrahedron::GetSize()
{
  return 4;
}
//-----------------------------------------------------------------------------
Node* Tetrahedron::GetNode(int node)
{
  if ( (node<0) || (node>=4) )
	 display->InternalError("Tetrahedron::GetNode()","Illegal node: %d",node);
  
  return nodes[node];
}
//-----------------------------------------------------------------------------
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
//-----------------------------------------------------------------------------
void Tetrahedron::CountCell(Node *node_list)
{
  node_list[nodes[0]->GetNodeNo()].nc += 1;
  node_list[nodes[1]->GetNodeNo()].nc += 1;
  node_list[nodes[2]->GetNodeNo()].nc += 1;
  node_list[nodes[3]->GetNodeNo()].nc += 1;
}
//-----------------------------------------------------------------------------
void Tetrahedron::AddCell(Node *node_list, int *current, int thiscell)
{
  int pos, n;
  n=nodes[0]->GetNodeNo(); pos=current[n]; node_list[n].neighbor_cells[pos]=thiscell; current[n]+=1;
  n=nodes[1]->GetNodeNo(); pos=current[n]; node_list[n].neighbor_cells[pos]=thiscell; current[n]+=1;
  n=nodes[2]->GetNodeNo(); pos=current[n]; node_list[n].neighbor_cells[pos]=thiscell; current[n]+=1;
  n=nodes[3]->GetNodeNo(); pos=current[n]; node_list[n].neighbor_cells[pos]=thiscell; current[n]+=1;
}
//-----------------------------------------------------------------------------
void Tetrahedron::AddNodes(int exclude_node, int *new_nodes, int *pos)
{
  int n;

  if ( (n = nodes[0]->GetNodeNo()) != exclude_node )
	 if ( !contains(new_nodes,*pos,n) ) new_nodes[(*pos)++] = n;
  if ( (n = nodes[1]->GetNodeNo()) != exclude_node )
	 if ( !contains(new_nodes,*pos,n) )	new_nodes[(*pos)++] = n;
  if ( (n = nodes[2]->GetNodeNo()) != exclude_node )
	 if ( !contains(new_nodes,*pos,n) ) new_nodes[(*pos)++] = n;
  if ( (n = nodes[3]->GetNodeNo()) != exclude_node )
	 if ( !contains(new_nodes,*pos,n) ) new_nodes[(*pos)++] = n;
}
//-----------------------------------------------------------------------------
void Tetrahedron::ComputeCellNeighbors(Node *node_list, int thiscell)
{
  // Although tetrahedrons on the boundary have only 3 neighbors, allocate
  // for 4 neighbors (data will be moved to another location anyway).
  if ( !neighbor_cells )
	 neighbor_cells = new int[4];
  
  int c;

  if ( node_list[nodes[0]->GetNodeNo()].CommonCell(&node_list[nodes[1]->GetNodeNo()],
						   &node_list[nodes[2]->GetNodeNo()],thiscell,&c) ){
	 neighbor_cells[nc] = c;
	 nc += 1;
  }
  if ( node_list[nodes[0]->GetNodeNo()].CommonCell(&node_list[nodes[1]->GetNodeNo()],
						   &node_list[nodes[3]->GetNodeNo()],thiscell,&c) ){	 
	 neighbor_cells[nc] = c;
	 nc += 1;
  }
  if ( node_list[nodes[0]->GetNodeNo()].CommonCell(&node_list[nodes[2]->GetNodeNo()],
						   &node_list[nodes[3]->GetNodeNo()],thiscell,&c) ){
	 neighbor_cells[nc] = c;
	 nc += 1;
  }
  if ( node_list[nodes[1]->GetNodeNo()].CommonCell(&node_list[nodes[2]->GetNodeNo()],
						   &node_list[nodes[3]->GetNodeNo()],thiscell,&c) ){
	 neighbor_cells[nc] = c;
	 nc += 1;
  }

}
//-----------------------------------------------------------------------------
