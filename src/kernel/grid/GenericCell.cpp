// Copyright (C) 2002 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#include <dolfin/dolfin_log.h>
#include <dolfin/Point.h>
#include <dolfin/Grid.h>
#include <dolfin/Cell.h>
#include <dolfin/GenericCell.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
GenericCell::GenericCell()
{
  grid = 0;
  _id = -1;
  _parent = 0;
  _marker = 0;
}
//-----------------------------------------------------------------------------
GenericCell::~GenericCell()
{
  if ( _marker )
    delete _marker;
  _marker = 0;
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
Node* GenericCell::node(int i) const
{
  return cn(i);
}
//-----------------------------------------------------------------------------
Edge* GenericCell::edge(int i) const
{
  return ce(i);
}
//-----------------------------------------------------------------------------
Cell* GenericCell::neighbor(int i) const
{
  return cc(i);
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
Point GenericCell::coord(int i) const
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
void GenericCell::mark(Cell* cell)
{
  if ( !grid )
    dolfin_error("You cannot mark a cell that does not belong to grid.");
 
  grid->mark(cell);
}
//-----------------------------------------------------------------------------
int GenericCell::setID(int id, Grid* grid)
{
  this->grid = grid;
  return _id = id;
}
//-----------------------------------------------------------------------------
void GenericCell::setParent(Cell* parent)
{
  // Set parent cell: a cell is parent if the current cell is created through 
  // refinement of the parent cell. 
  this->_parent = parent;
}
//-----------------------------------------------------------------------------
void GenericCell::setChild(Cell* child)
{
  // Set the child cell if Cell not already contains the child cell: a cell 
  // is child if it is created through refinement of the current cell.  
  if ( !children.contains(child) )
    children.add(child);
}
//-----------------------------------------------------------------------------
bool GenericCell::neighbor(GenericCell* cell) const
{
  // Two cells are neighbors if they have a common edge or if they are
  // the same cell, i.e. if they have 2 or 3 common nodes.

  if ( !cell )
    return false;

  int count = 0;
  for (int i = 0; i < cn.size(); i++)
    for (int j = 0; j < cell->cn.size(); j++)
      if ( cn(i) == cell->cn(j) )
	count++;
  
  return count >= 2;
}
//-----------------------------------------------------------------------------
void GenericCell::createEdge(Node* n0, Node* n1)
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
    edge = grid->createEdge(n0, n1);


  // Add the edge at the first empty position
  ce.add(edge);
}
//-----------------------------------------------------------------------------
void GenericCell::createFace(Edge* e0, Edge* e1, Edge* e2)
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
    face = grid->createFace(e0, e1, e2);

  // Add the face at the first empty position
  cf.add(face);
}
//-----------------------------------------------------------------------------
Edge* GenericCell::findEdge(Node* n0, Node* n1)
{
  for (Array<Edge*>::Iterator e(ce); !e.end(); ++e)
    if ( *e )
      if ( (*e)->equals(n0, n1) )
	return *e;

  return 0;
}
//-----------------------------------------------------------------------------
Face* GenericCell::findFace(Edge* e0, Edge* e1, Edge* e2)
{
  for (Array<Face*>::Iterator f(cf); !f.end(); ++f)
    if ( *f )
      if ( (*f)->equals(e0, e1, e2) )
	return *f;

  return 0;
}
//-----------------------------------------------------------------------------
CellMarker& GenericCell::marker() const
{
  dolfin_assert(_marker);

  return *_marker;
}
//-----------------------------------------------------------------------------
