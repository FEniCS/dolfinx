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
int Cell::noNodes()
{
  if ( c )
	 return c->noNodes();

  return 0;
}
//-----------------------------------------------------------------------------
int Cell::noEdges()
{
  if ( c )
	 return c->noEdges();

  return 0;
}
//-----------------------------------------------------------------------------
int Cell::noFaces()
{
  if ( c )
	 return c->noFaces();

  return 0;
}
//-----------------------------------------------------------------------------
int Cell::noBoundaries()
{
  if ( c )
	 return c->noBoundaries();

  return 0;
}
//-----------------------------------------------------------------------------
int Cell::id() const
{
  return _id;
}
//-----------------------------------------------------------------------------
Cell::Type Cell::type() const
{
  if ( c )
	 return c->type();

  return NONE;
}  
//-----------------------------------------------------------------------------
void Cell::set(Node *n0, Node *n1, Node *n2)
{
  if ( cn.size() != 3 ) {
	 // FIXME: Temporary until we fix the log system
	 cout << "Wrong number of nodes for this cell type." << endl;
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
	 cout << "Wrong number of nodes for this cell type." << endl;
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
  
  switch (type){
  case TRIANGLE:
	 c = new Triangle();
	 break;
  case TETRAHEDRON:
	 c = new Tetrahedron();
	 break;
  default:
	 // FIXME: Temporary until we fix the log system
	 cout << "Unknown cell type" << endl;
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
namespace dolfin {

  //---------------------------------------------------------------------------
  std::ostream& operator << (std::ostream& output, const Cell& cell)
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
		cout << "Unknown cell type" << endl;
		exit(1);
	 }	 

	 return output;
  }
  //---------------------------------------------------------------------------

}
//-----------------------------------------------------------------------------
