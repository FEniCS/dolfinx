#include <dolfin/Node.h>
#include <dolfin/GenericCell.h>
#include <dolfin/Triangle.h>
#include <dolfin/Tetrahedron.h>
#include <dolfin/Cell.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
Cell::Cell()
{
  _id = -1;
  c = 0;
  _marker = UNMARKED;
}
//-----------------------------------------------------------------------------
Cell::Cell(Node &n0, Node &n1, Node &n2)
{
  _id = -1;

  c = 0;
  init(TRIANGLE);

  cn(0) = &n0;
  cn(1) = &n1;
  cn(2) = &n2;

  _marker = UNMARKED;
}
//-----------------------------------------------------------------------------
Cell::Cell(Node &n0, Node &n1, Node &n2, Node &n3)
{
  _id = -1;

  c = 0;
  init(TETRAHEDRON);

  cn(0) = &n0;
  cn(1) = &n1;
  cn(2) = &n2;
  cn(3) = &n3;

  _marker = UNMARKED;
}
//-----------------------------------------------------------------------------
Cell::~Cell()
{
  if ( c )
	 delete c;
}
//-----------------------------------------------------------------------------
GenericCell* Cell::operator->() const
{
  return c;
}
//-----------------------------------------------------------------------------
int Cell::noNodes() const
{
  if ( c )
	 return c->noNodes();

  return 0;
}
//-----------------------------------------------------------------------------
int Cell::noEdges() const
{
  if ( c )
	 return c->noEdges();

  return 0;
}
//-----------------------------------------------------------------------------
int Cell::noFaces() const
{
  if ( c )
	 return c->noFaces();

  return 0;
}
//-----------------------------------------------------------------------------
int Cell::noBound() const
{
  if ( c )
	 return c->noBound();

  return 0;
}
//-----------------------------------------------------------------------------
Node* Cell::node(int i) const
{
  return cn(i);
}
//-----------------------------------------------------------------------------
Point Cell::coord(int i) const
{
  return cn(i)->coord();
}
//-----------------------------------------------------------------------------
Cell::Type Cell::type() const
{
  if ( c )
	 return c->type();

  return NONE;
}
//-----------------------------------------------------------------------------
int Cell::noCellNeighbors() const
{
  return cc.size();
}
//-----------------------------------------------------------------------------
int Cell::noNodeNeighbors() const
{
  return cn.size();
}
//-----------------------------------------------------------------------------
int Cell::id() const
{
  return _id;
}
//-----------------------------------------------------------------------------
int Cell::nodeID(int i) const
{
  return cn(i)->id();
}
//-----------------------------------------------------------------------------
int Cell::edgeID(int i) const
{
  // FIXME: Use logging system
  std::cout << "Cell::edgeID() is not implemented." << std::endl;
  exit(1);
}
//-----------------------------------------------------------------------------
void Cell::mark()
{
  _marker = MARKED;
}
//-----------------------------------------------------------------------------
bool Cell::marked() const
{
  return _marker == MARKED;
}
//-----------------------------------------------------------------------------
void Cell::mark(Marker marker)
{
  _marker = marker;
}
//-----------------------------------------------------------------------------
Cell::Marker Cell::marker() const
{
  return _marker;
}
//-----------------------------------------------------------------------------
void Cell::set(Node *n0, Node *n1, Node *n2)
{
  if ( cn.size() != 3 ) {
	 // FIXME: Temporary until we fix the log system
	 std::cout << "Wrong number of nodes for this cell type." << std::endl;
	 exit(1);
  }

  cn(0) = n0;
  cn(1) = n1;
  cn(2) = n2;
}
//-----------------------------------------------------------------------------
void Cell::set(Node *n0, Node *n1, Node *n2, Node *n3)
{
  if ( cn.size() != 4 ) {
	 // FIXME: Temporary until we fix the log system
	 std::cout << "Wrong number of nodes for this cell type." << std::endl;
	 exit(1);
  }

  cn(0) = n0;
  cn(1) = n1;
  cn(2) = n2;
  cn(3) = n3;
}
//-----------------------------------------------------------------------------
void Cell::setID(int id)
{
  _id = id;
}
//-----------------------------------------------------------------------------
void Cell::init(Type type)
{
  if ( c )
	 delete c;
  
  switch (type) {
  case TRIANGLE:
	 c = new Triangle();
	 break;
  case TETRAHEDRON:
	 c = new Tetrahedron();
	 break;
  default:
	 // FIXME: Temporary until we fix the log system
	 std::cout << "Unknown cell type" << std::endl;
	 exit(1);
  }
  
  cn.init(noNodes());
}
//-----------------------------------------------------------------------------
bool Cell::neighbor(Cell &cell)
{
  if ( c )
	 return c->neighbor(cn,cell);

  return false;
}
//-----------------------------------------------------------------------------
// Additional operators
//-----------------------------------------------------------------------------
std::ostream& dolfin::operator << (std::ostream& output, const Cell& cell)
{
  switch ( cell.type() ){
  case Cell::TRIANGLE:
	 output << "[Cell (triangle) with nodes ( ";
	 for (NodeIterator n(cell); !n.end(); ++n)
		output << n->id() << " ";
	 output << "]";
	 break;
  case Cell::TETRAHEDRON:
	 output << "[Cell (tetrahedron) with nodes ( ";
	 for (NodeIterator n(cell); !n.end(); ++n)
		output << n->id() << " ";
	 output << "]";
	 break;
  default:
	 // FIXME: Temporary until we fix the log system
	 std::cout << "Unknown cell type" << std::endl;
	 exit(1);
  }	 
  
  return output;
}
//-----------------------------------------------------------------------------
