// Copyright (C) 2002 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#include <dolfin/dolfin_log.h>
#include <dolfin/Point.h>
#include <dolfin/Mesh.h>
#include <dolfin/Cell.h>
#include <dolfin/CellRefData.h>
#include <dolfin/GenericCell.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
GenericCell::GenericCell()
{
  mesh = 0;
  _id = -1;
  _parent = 0;
  rd = 0;
}
//-----------------------------------------------------------------------------
GenericCell::~GenericCell()
{
  if ( rd )
    delete rd;
  rd = 0;
}
//-----------------------------------------------------------------------------
int GenericCell::id() const
{
  return _id;
}
//-----------------------------------------------------------------------------
int GenericCell::noCellNeighbors() const
{
  return cc.size();
}
//-----------------------------------------------------------------------------
int GenericCell::noNodeNeighbors() const
{
  return cn.size();
}
//-----------------------------------------------------------------------------
int GenericCell::noChildren() const
{
  return children.size();
}
//-----------------------------------------------------------------------------
Node& GenericCell::node(int i) const
{
  return *cn(i);
}
//-----------------------------------------------------------------------------
Edge& GenericCell::edge(int i) const
{
  return *ce(i);
}
//-----------------------------------------------------------------------------
Face& GenericCell::face(int i) const
{
  return *cf(i);
}
//-----------------------------------------------------------------------------
Cell& GenericCell::neighbor(int i) const
{
  return *cc(i);
}
//-----------------------------------------------------------------------------
Cell* GenericCell::parent() const
{
  return _parent;
}
//-----------------------------------------------------------------------------
Cell* GenericCell::child(int i) const
{
  return children(i);
}
//-----------------------------------------------------------------------------
Point& GenericCell::coord(int i) const
{
  return cn(i)->coord();
}
//-----------------------------------------------------------------------------
Point GenericCell::midpoint() const
{
  Point p;

  for (Array<Node*>::Iterator n(cn); !n.end(); ++n)
    p += (*n)->coord();

  p /= real(cn.size());

  return p;
}
//-----------------------------------------------------------------------------
int GenericCell::nodeID(int i) const
{
  return cn(i)->id();
}
//-----------------------------------------------------------------------------
void GenericCell::mark()
{
  marker() = Cell::marked_for_reg_ref;
}
//-----------------------------------------------------------------------------
int GenericCell::setID(int id, Mesh& mesh)
{
  this->mesh = &mesh;
  return _id = id;
}
//-----------------------------------------------------------------------------
void GenericCell::setMesh(Mesh& mesh)
{
  this->mesh = &mesh;
}
//-----------------------------------------------------------------------------
void GenericCell::setParent(Cell& parent)
{
  // Set parent cell: a cell is parent if the current cell is created
  // through refinement of the parent cell.
  this->_parent = &parent;
}
//-----------------------------------------------------------------------------
void GenericCell::removeParent()
{
  // Remove parent cell
  this->_parent = 0;
}
//-----------------------------------------------------------------------------
void GenericCell::initChildren(int n)
{
  children.init(n);
  children.reset();
}
//-----------------------------------------------------------------------------
void GenericCell::addChild(Cell& child)
{
  // Set the child cell: a cell is child if it is created through
  // refinement of the current cell.
  children.add(&child);
}
//-----------------------------------------------------------------------------
void GenericCell::removeChild(Cell& child)
{
  // Remove the child cell
  children.remove(&child);
  children.resize();
}
//-----------------------------------------------------------------------------
bool GenericCell::neighbor(GenericCell& cell) const
{
  // Two cells are neighbors if they have a common edge or if they are
  // the same cell, i.e. if they have 2 or 3 common nodes.

  int count = 0;
  for (int i = 0; i < cn.size(); i++)
    for (int j = 0; j < cell.cn.size(); j++)
      if ( cn(i) == cell.cn(j) )
	count++;
  
  return count >= 2;
}
//-----------------------------------------------------------------------------
bool GenericCell::haveNode(Node& node) const
{
  for (Array<Node*>::Iterator n(cn); !n.end(); ++n)
    if ( *n == &node )
      return true;
  return false;
}
//-----------------------------------------------------------------------------
bool GenericCell::haveEdge(Edge& edge) const
{
  for (Array<Edge*>::Iterator e(ce); !e.end(); ++e)
    if ( *e == &edge )
      return true;
  return false;
}
//-----------------------------------------------------------------------------
void GenericCell::createEdge(Node& n0, Node& n1)
{
  Edge* edge = 0;

  // Check neighbor cells if an edge already exists between the two nodes
  for (Array<Cell*>::Iterator c(cc); !c.end(); ++c) {
    edge = (*c)->findEdge(n0, n1);
    if ( edge )
      break;
  }

  // Create the new edge if it doesn't exist
  if ( !edge )
    edge = &mesh->createEdge(n0, n1);


  // Add the edge at the first empty position
  ce.add(edge);
}
//-----------------------------------------------------------------------------
void GenericCell::createFace(Edge& e0, Edge& e1, Edge& e2)
{
  Face* face = 0;
  
  // Check neighbor cells if the face already exists
  for (Array<Cell*>::Iterator c(cc); !c.end(); ++c) {
    face = (*c)->findFace(e0, e1, e2);
    if ( face )
      break;
  }

  // Create the new face if it doesn't exist
  if ( !face )
    face = &mesh->createFace(e0, e1, e2);

  // Add the face at the first empty position
  cf.add(face);
}
//-----------------------------------------------------------------------------
Node* GenericCell::findNode(const Point& p) const
{
  for (Array<Node*>::Iterator n(cn); !n.end(); ++n)
    if ( (*n)->dist(p) < DOLFIN_EPS )
      return *n;

  return 0;
}
//-----------------------------------------------------------------------------
Edge* GenericCell::findEdge(Node& n0, Node& n1)
{
  for (Array<Edge*>::Iterator e(ce); !e.end(); ++e)
    if ( *e )
      if ( (*e)->equals(n0, n1) )
	return *e;

  return 0;
}
//-----------------------------------------------------------------------------
Face* GenericCell::findFace(Edge& e0, Edge& e1, Edge& e2)
{
  for (Array<Face*>::Iterator f(cf); !f.end(); ++f)
    if ( *f )
      if ( (*f)->equals(e0, e1, e2) )
	return *f;

  return 0;
}
//-----------------------------------------------------------------------------
Face* GenericCell::findFace(Edge& e0, Edge& e1)
{
  for (Array<Face*>::Iterator f(cf); !f.end(); ++f)
    if ( *f )
      if ( (*f)->equals(e0, e1) )
	return *f;

  return 0;
}
//-----------------------------------------------------------------------------
void GenericCell::initMarker()
{
  if ( !rd )
    rd = new CellRefData();
  dolfin_assert(rd);
}
//-----------------------------------------------------------------------------
Cell::Marker& GenericCell::marker()
{
  initMarker();
  return rd->marker;
}
//-----------------------------------------------------------------------------
Cell::Status& GenericCell::status()
{
  initMarker();
  return rd->status;
}
//-----------------------------------------------------------------------------
