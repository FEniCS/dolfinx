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
  _no_children = 0;

  _marked_for_re_use = true;

  _refined_by_face_rule = false;
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

  _marker = MARKED_FOR_NO_REFINEMENT;
  _status = UNREFINED;

  _no_marked_edges = 0;
  _no_children = 0;

  _marked_for_re_use = true;

  _refined_by_face_rule = false;
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

  _marker = MARKED_FOR_NO_REFINEMENT;
  _status = UNREFINED;

  _no_marked_edges = 0;
  _no_children = 0;

  _marked_for_re_use = true;

  _refined_by_face_rule = false;
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
int Cell::noChildren() const
{
  return _no_children;
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
Cell* Cell::child(int i) const
{
  return children(i);
}
//-----------------------------------------------------------------------------
void Cell::addChild(Cell* child) 
{
  children.add(child);
  _no_children++;
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
void Cell::setEdge(Edge* e, int i)
{
  ce(i) = e;
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
void Cell::setMarkedForReUse(bool re_use)
{
  _marked_for_re_use = re_use;
}
//-----------------------------------------------------------------------------
bool Cell::markedForReUse()
{
  return _marked_for_re_use;
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
void Cell::markEdge(int edge)
{
  if (!ce(edge)->marked()){
    ce(edge)->mark();
    ce(edge)->setRefinedByCell(this);
    _no_marked_edges++;
  }
}	 
//-----------------------------------------------------------------------------
void Cell::unmarkEdge(int edge)
{
  if (ce(edge)->marked()){
    ce(edge)->unmark();
    _no_marked_edges--;
  }	 
}
//-----------------------------------------------------------------------------
int Cell::noMarkedEdges()
{
  return _no_marked_edges;
}	 
//-----------------------------------------------------------------------------
void Cell::refineByFaceRule(bool refined_by_face_rule)
{
  _refined_by_face_rule = refined_by_face_rule;
}	 
//-----------------------------------------------------------------------------
bool Cell::refinedByFaceRule()
{
  return _refined_by_face_rule;
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
	if (edge(i)->marked()){
	  for (int j=0;j<noNodes();j++){
	    if (edge(i)->node(0)->id() == node(j)->id()) marked_node[j] = true; 	
	    if (edge(i)->node(1)->id() == node(j)->id()) marked_node[j] = true; 	
	  }
	}
      }
      for (int i=0;i<noNodes();i++){
	if (marked_node[i] == false) return true;
      }
      return false;
      break;
    case 3:
      marked_node[0] = marked_node[1] = marked_node[2] = marked_node[3] = false;
      for (int i=0;i<noEdges();i++){
	if (edge(i)->marked()){
	  for (int j=0;j<noNodes();j++){
	    if (edge(i)->node(0)->id() == node(j)->id()) marked_node[j] = true; 	
	    if (edge(i)->node(1)->id() == node(j)->id()) marked_node[j] = true; 	
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
