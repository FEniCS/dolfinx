#include <dolfin/Node.h>
#include <dolfin/GenericCell.h>
#include <dolfin/Display.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
Node::Node()
{
  neighbor_nodes = 0;
  neighbor_cells = 0;
  nn = 0;
  nc = 0;

  _id = -1;
}
//-----------------------------------------------------------------------------
Node::~Node()
{
  Clear();
}
//-----------------------------------------------------------------------------
void Node::set(real x, real y, real z)
{
  p.x = x;
  p.y = y;
  p.z = z;
}
//-----------------------------------------------------------------------------
int Node::id() const
{
  return _id;
}
//-----------------------------------------------------------------------------
Point Node::coord() const
{
  return p;
}
//-----------------------------------------------------------------------------
void Node::Clear()
{
  if ( neighbor_nodes )
	 delete [] neighbor_nodes;
  neighbor_nodes = 0;

  if ( neighbor_cells )
	 delete [] neighbor_cells;
  neighbor_cells = 0;
  
  nn = 0;
  nc = 0;
}
//-----------------------------------------------------------------------------
int Node::GetNoCellNeighbors()
{
  return ( nc );
}
//-----------------------------------------------------------------------------
int Node::GetCellNeighbor(int i)
{
  if ( (i<0) || (i>=nc) )
	 display->InternalError("Node::GetCellNeighbor()","Illegal index: %d",i);

  return ( neighbor_cells[i] );
}
//-----------------------------------------------------------------------------
int Node::GetNoNodeNeighbors()
{
  return ( nn );
}
//-----------------------------------------------------------------------------
int Node::GetNodeNeighbor(int i)
{
  if ( (i<0) || (i>=nn) )
	 display->InternalError("Node::GetNodeNeighbor()","Illegal index: %d",i);

  return ( neighbor_nodes[i] );
}
//-----------------------------------------------------------------------------
void Node::SetNodeNo(int nn)
{
  global_node_number = nn;
}
//-----------------------------------------------------------------------------
void Node::SetCoord(float x, float y, float z)
{
  p.x = x;
  p.y = y;
  p.z = z;
}
//-----------------------------------------------------------------------------
int Node::GetNodeNo()
{
  return global_node_number;
}
//-----------------------------------------------------------------------------
real Node::GetCoord(int i)
{
  switch (i){
  case 0:
	 return ( real(p.x) );
	 break;
  case 1:
	 return ( real(p.y) );
	 break;
  case 2:
	 return ( real(p.z) );
	 break;
  default:
	 display->InternalError("Node::GetCoord()","Illegal index: %d",i);
  }

  return 0.0;
}
//-----------------------------------------------------------------------------
Point* Node::GetCoord()
{
  return ( &p );
}
//-----------------------------------------------------------------------------
int Node::setID(int id)
{
  return _id = id;
}
//-----------------------------------------------------------------------------
void Node::AllocateForNeighborCells()
{
  if ( neighbor_cells )
	 delete [] neighbor_cells;
  neighbor_cells = new int[nc];
}
//-----------------------------------------------------------------------------
bool Node::CommonCell(Node *n, int thiscell, int *cellnumber)
{
  int c1, c2;
  
  for (int i=0;i<nc;i++){
	 c1 = neighbor_cells[i];
	 if ( c1 == thiscell )
		continue;
	 for (int j=0;j<(n->nc);j++){
		c2 = n->neighbor_cells[j];
		if ( c1==c2 ){
		  *cellnumber = c1;
		  return true;
		}
	 }
  }
  
  return false;
}
//-----------------------------------------------------------------------------
bool Node::CommonCell(Node *n1, Node *n2, int thiscell, int *cellnumber)
{
  int c1, c2, c3;

  for (int i=0;i<nc;i++){
	 c1 = neighbor_cells[i];
	 if ( c1 == thiscell )
		continue;
	 for (int j=0;j<(n1->nc);j++){
		c2 = n1->neighbor_cells[j];
		for (int k=0;k<(n2->nc);k++){
		  c3 = n2->neighbor_cells[k];
		  if ( (c1==c2) && (c2==c3) ){
			 *cellnumber = c1;
			 return true;
		  }
		}
	 }
  }
  
  return false;
}
//-----------------------------------------------------------------------------
int Node::GetMaxNodeNeighbors(GenericCell **cell_list)
{
  int max = 0;

  // Add the node itself
  max = 1;
  
  // Add all other nodes in cell neighbors
  for (int i=0;i<nc;i++)
  	 max += (cell_list[neighbor_cells[i]]->GetSize() - 1);
  
  return ( max );
}
//-----------------------------------------------------------------------------
void Node::ComputeNodeNeighbors(GenericCell **cell_list, int thisnode, int *tmp)
{
  // tmp is allocated for the maximum number of node neighbors
  
  int pos = 0;

  // Add the node itself
  tmp[0] = thisnode;
  pos = 1;
  
  // Add all other nodes in cell neighbors
  for (int i=0;i<nc;i++)
	 cell_list[neighbor_cells[i]]->AddNodes(thisnode,tmp,&pos);

  // Allocate memory for node neighbors and put the neighbors in the list
  nn = pos;
  neighbor_nodes = new int[nn];
  for (int i=0;i<nn;i++)
	 neighbor_nodes[i] = tmp[i];
}
//-----------------------------------------------------------------------------
namespace dolfin {

  //---------------------------------------------------------------------------
  std::ostream& operator << (std::ostream& output, const Node& node)
  {
	 int id = node.id();
	 Point p = node.coord();
	 
	 output << "[ Node: id = " << id
			  << " x = (" << p.x << "," << p.y << "," << p.z << ") ]";

	 return output;
  }
  //---------------------------------------------------------------------------

}
//-----------------------------------------------------------------------------
