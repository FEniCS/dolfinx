#include <dolfin/Node.h>
#include <dolfin/GenericCell.h>
#include <dolfin/Triangle.h>
#include <dolfin/Tetrahedron.h>
#include <dolfin/Cell.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
Cell::Cell()
{
  c = 0;
}
//-----------------------------------------------------------------------------
Cell::~Cell()
{
  if ( c )
	 delete c;
}
//-----------------------------------------------------------------------------
int Cell::id() const
{
  if ( c )
	 return c->id();

  return -1;
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
  if ( !c ){
	 // FIXME: Temporary until we fix the log system
	 cout << "Cannot set cell nodes for unitialised cell." << endl;
	 exit(1);
  }
  
  switch ( c->type() ){
  case TRIANGLE:
	 ( (Triangle *) c )->set(n0,n1,n2);
	 break;
  default:
	 // FIXME: Temporary until we fix the log system
	 cout << "Cannot set four nodes for this cell type." << endl;
	 exit(1);
  }
}
//-----------------------------------------------------------------------------
void Cell::set(Node *n0, Node *n1, Node *n2, Node *n3)
{
  if ( !c ){
	 cout << "Cannot set cell nodes for unitialised cell." << endl;
	 exit(1);
  }
  
  switch ( c->type() ){
  case TETRAHEDRON:
	 ( (Tetrahedron *) c )->set(n0,n1,n2,n3);
	 break;
  default:
	 // FIXME: Temporary until we fix the log system
	 cout << "Cannot set four nodes for this cell type." << endl;
	 exit(1);
  }
}
//-----------------------------------------------------------------------------
void Cell::setID(int id)
{
  if ( c )
	 c->setID(id);
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

}
//-----------------------------------------------------------------------------
namespace dolfin {

  //---------------------------------------------------------------------------
  std::ostream& operator << (std::ostream& output, const Cell& cell)
  {
	 switch ( cell.type() ){
	 case Cell::TRIANGLE:
		output << *( (Triangle *) cell.c );
		break;
	 case Cell::TETRAHEDRON:
		output << *( (Tetrahedron *) cell.c );
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
