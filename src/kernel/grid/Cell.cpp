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
  c = 0;

  // Remove?
  _marker = MARKED_FOR_NO_REFINEMENT;
  _status = UNREFINED;
  _no_marked_edges = 0;
  _marked_for_re_use = true;
  _refined_by_face_rule = false;
}
//-----------------------------------------------------------------------------
Cell::Cell(Node* n0, Node* n1, Node* n2)
{
  c = new Triangle(n0, n1, n2);

  // FIXME: Remove?
  _level = -1;
  _marker = MARKED_FOR_NO_REFINEMENT;
  _status = UNREFINED;
  _no_marked_edges = 0;
  _marked_for_re_use = true;
  _refined_by_face_rule = false;
}
//-----------------------------------------------------------------------------
Cell::Cell(Node* n0, Node* n1, Node* n2, Node* n3)
{
  c = new Tetrahedron(n0, n1, n2, n3);

  // FIXME: Remove?
  _level = -1;
  _marker = MARKED_FOR_NO_REFINEMENT;
  _status = UNREFINED;
  _no_marked_edges = 0;
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

  return none;
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
int Cell::noBoundaries() const
{
  if ( c )
    return c->noBoundaries();

  return 0;
}
//-----------------------------------------------------------------------------
int Cell::noCellNeighbors() const
{
  if ( c )
    return c->noCellNeighbors();

  return 0;
}
//-----------------------------------------------------------------------------
int Cell::noNodeNeighbors() const
{
  if ( c )
    return c->noNodeNeighbors();

  return 0;
}
//-----------------------------------------------------------------------------
Node* Cell::node(int i) const
{
  if ( c )
    return c->node(i);

  return 0;
}
//-----------------------------------------------------------------------------
Edge* Cell::edge(int i) const
{
  if ( c )
    return c->edge(i);

  return 0;
}
//-----------------------------------------------------------------------------
Cell* Cell::neighbor(int i) const
{
  if ( c )
    return c->neighbor(i);

  return 0;
}
//-----------------------------------------------------------------------------
Point Cell::coord(int i) const
{ 
  if ( c )
    return c->coord(i);

  Point p;
  return p;
}
//-----------------------------------------------------------------------------
Point Cell::midpoint() const
{
  if ( c )
    return c->midpoint();

  Point p;
  return p;
}
//-----------------------------------------------------------------------------
int Cell::nodeID(int i) const
{
  if ( c )
    return c->nodeID(i);
  
  return -1;
}
//-----------------------------------------------------------------------------
void Cell::mark()
{
  if ( !c )
    dolfin_error("You cannot mark an unspecified cell.");
  
  c->mark(this);
}
//-----------------------------------------------------------------------------
int Cell::setID(int id, Grid* grid)
{
  dolfin_assert(c);
  return c->setID(id, grid);
}
//-----------------------------------------------------------------------------
void Cell::set(Node* n0, Node* n1, Node* n2)
{
  if ( c )
    delete c;
  
  c = new Triangle(n0, n1, n2);
}
//-----------------------------------------------------------------------------
void Cell::set(Node* n0, Node* n1, Node* n2, Node* n3)
{
  if ( c )
    delete c;

  c = new Tetrahedron(n0, n1, n2, n3);
}
//-----------------------------------------------------------------------------
bool Cell::neighbor(Cell& cell)
{
  if ( c )
    return c->neighbor(cell.c);
  
  return false;
}
//-----------------------------------------------------------------------------
void Cell::createEdges()
{
  dolfin_assert(c);
  c->createEdges();
}
//-----------------------------------------------------------------------------
void Cell::createFaces()
{
  dolfin_assert(c);
  c->createFaces();
}
//-----------------------------------------------------------------------------
void Cell::createEdge(Node* n0, Node* n1)
{
  dolfin_assert(c);
  c->createEdge(n0, n1);
}
//-----------------------------------------------------------------------------
void Cell::createFace(Edge* e0, Edge* e1, Edge* e2)
{
  dolfin_assert(c);
  c->createFace(e0, e1, e2);
}
//-----------------------------------------------------------------------------
Edge* Cell::findEdge(Node* n0, Node* n1)
{
  dolfin_assert(c);
  return c->findEdge(n0, n1);
}
//-----------------------------------------------------------------------------
Face* Cell::findFace(Edge* e0, Edge* e1, Edge* e2)
{
  dolfin_assert(c);
  return c->findFace(e0, e1, e2);
}
//-----------------------------------------------------------------------------
// Additional operators
//-----------------------------------------------------------------------------
dolfin::LogStream& dolfin::operator<<(LogStream& stream, const Cell& cell)
{
  switch ( cell.type() ){
  case Cell::triangle:
    stream << "[Cell (triangle) with nodes ( ";
    for (NodeIterator n(cell); !n.end(); ++n)
      stream << n->id() << " ";
    stream << "]";
    break;
  case Cell::tetrahedron:
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






/*
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
  case triangle:
    return true;
    break;
  case tetrahedron:
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
*/
