// Copyright (C) 2003 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#include <dolfin/dolfin_log.h>
#include <dolfin/Node.h>
#include <dolfin/Edge.h>
#include <dolfin/GenericCell.h>
#include <dolfin/Triangle.h>
#include <dolfin/Tetrahedron.h>
#include <dolfin/Cell.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
Cell::Cell()
{
  _level = -1;
  _id = -1;
  c = 0;
  _marker = MARKED_FOR_NO_REFINEMENT;
  _status = UNREFINED;

  _no_marked_edges = 0;
}
//-----------------------------------------------------------------------------
Cell::Cell(Node &n0, Node &n1, Node &n2)
{
  _level = -1;
  _id = -1;

  c = 0;
  init(TRIANGLE);

  cn(0) = &n0;
  cn(1) = &n1;
  cn(2) = &n2;

  ce(0)->set(&n0,&n1);
  ce(1)->set(&n0,&n2);
  ce(2)->set(&n1,&n2);

  _marker = MARKED_FOR_NO_REFINEMENT;
  _status = UNREFINED;

  _no_marked_edges = 0;
}
//-----------------------------------------------------------------------------
Cell::Cell(Node &n0, Node &n1, Node &n2, Node &n3)
{
  _level = -1;
  _id = -1;

  c = 0;
  init(TETRAHEDRON);

  cn(0) = &n0;
  cn(1) = &n1;
  cn(2) = &n2;
  cn(3) = &n3;

  ce(0)->set(&n0,&n1);
  ce(1)->set(&n0,&n2);
  ce(2)->set(&n0,&n3);
  ce(3)->set(&n1,&n2);
  ce(4)->set(&n1,&n3);
  ce(5)->set(&n2,&n3);

  _marker = MARKED_FOR_NO_REFINEMENT;
  _status = UNREFINED;

  _no_marked_edges = 0;
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
Edge* Cell::edge(int i) const
{
  return ce(i);
}
//-----------------------------------------------------------------------------
Cell* Cell::neighbor(int i) const
{
  return cc(i);
}
//-----------------------------------------------------------------------------
Cell* Cell::son(int i) const
{
  return children(i);
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
int Cell::level() const
{
  return _level;
}
//-----------------------------------------------------------------------------
int Cell::nodeID(int i) const
{
  return cn(i)->id();
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
void Cell::setStatus(Status status)
{
  _status = status;
}
//-----------------------------------------------------------------------------
Cell::Status Cell::status() const
{
  return _status;
}
//-----------------------------------------------------------------------------
void Cell::set(Node *n0, Node *n1, Node *n2)
{
  if ( cn.size() != 3 )
	 dolfin_error("Wrong number of nodes for this cell type.");

  cn(0) = n0;
  cn(1) = n1;
  cn(2) = n2;

  ce(0)->set(n0,n1);
  ce(1)->set(n0,n2);
  ce(2)->set(n1,n2);
}
//-----------------------------------------------------------------------------
void Cell::set(Node *n0, Node *n1, Node *n2, Node *n3)
{
  if ( cn.size() != 4 )
	 dolfin_error("Wrong number of nodes for this cell type.");

  cn(0) = n0;
  cn(1) = n1;
  cn(2) = n2;
  cn(3) = n3;

  Edge e1(n0,n1);
  Edge e2(n0,n2);
  Edge e3(n0,n3);
  Edge e4(n1,n1);
  Edge e5(n1,n2);
  Edge e6(n2,n3);

  ce(0) = &e1;
  ce(1) = &e2;
  ce(2) = &e3;
  ce(3) = &e4;
  ce(4) = &e5;
  ce(5) = &e6;
}
//-----------------------------------------------------------------------------
void Cell::setID(int id)
{
  _id = id;
}
//-----------------------------------------------------------------------------
void Cell::setLevel(int level)
{
  _level = level;
}
//-----------------------------------------------------------------------------
void Cell::mark_edge(int edge)
{
  ce(edge)->mark();

  if (!ce(edge)->marked()) _no_marked_edges++;
}	 
//-----------------------------------------------------------------------------
void Cell::unmark_edge(int edge)
{
  ce(edge)->unmark();

  if (ce(edge)->marked()) _no_marked_edges--;
}	 
//-----------------------------------------------------------------------------
int Cell::noMarkedEdges()
{
  return _no_marked_edges;
}	 
//-----------------------------------------------------------------------------
bool Cell::markedEdgesOnSameFace()
{
  bool marked_node[4];
  switch (type()) {
  case TRIANGLE:
    return true;
    break;
  case TETRAHEDRON:
    switch (_no_marked_edges) {
    case 1:
      return true;
      break;
    case 2:
      marked_node[0] = marked_node[1] = marked_node[2] = marked_node[3] = false;
      for (int i=0;i<noEdges();i++){
	if (ce(i)->marked()){
	  for (int j=0;j<noNodes();j++){
	    if (ce(i)->node(0)->id() == cn(j)->id()) marked_node[j] = true; 	
	    if (ce(i)->node(1)->id() == cn(j)->id()) marked_node[j] = true; 	
	  }
	}
      }
      for (int i=0;i<noNodes();i++){
	if (marked_node[i] = false) return true;
      }
      return false;
      break;
    case 3:
      marked_node[0] = marked_node[1] = marked_node[2] = marked_node[3] = false;
      for (int i=0;i<noEdges();i++){
	if (ce(i)->marked()){
	  for (int j=0;j<noNodes();j++){
	    if (ce(i)->node(0)->id() == cn(j)->id()) marked_node[j] = true; 	
	    if (ce(i)->node(1)->id() == cn(j)->id()) marked_node[j] = true; 	
	  }
	}
      }
      for (int i=0;i<noNodes();i++){
	if (marked_node[i] = false) return true;
      }
      return false;
      break;
    case 4:
      return true;
      break;
    case 5:
      return true;
      break;
    case 6:
      return true;
      break;
    default:
      dolfin_error("wrong number of marked edges");
    }
    break;
  default:
    dolfin_error("Unknown cell type.");
  }
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
	 dolfin_error("Unknown cell type.");
  }
  
  cn.init(noNodes());
  ce.init(noEdges());
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
dolfin::LogStream& dolfin::operator<<(LogStream& stream, const Cell& cell)
{
  switch ( cell.type() ){
  case Cell::TRIANGLE:
	 stream << "[Cell (triangle) with nodes ( ";
	 for (NodeIterator n(cell); !n.end(); ++n)
		stream << n->id() << " ";
	 stream << "]";
	 break;
  case Cell::TETRAHEDRON:
	 stream << "[Cell (tetrahedron) with nodes ( ";
	 for (NodeIterator n(cell); !n.end(); ++n)
		stream << n->id() << " ";
	 stream << "]";
	 break;
  default:
	 dolfin_error("Unknown cell type");
  }	 
  
  return stream;
}
//-----------------------------------------------------------------------------
