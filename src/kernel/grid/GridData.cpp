// Copyright (C) 2002 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#include <cmath>
#include <dolfin/Node.h>
#include <dolfin/Triangle.h>
#include <dolfin/Tetrahedron.h>
#include <dolfin/Cell.h>
#include <dolfin/GridData.h>
#include <dolfin/constants.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
Node* GridData::createNode(int level)
{ 
  int id;
  Node *n = nodes.create(&id);
  n->setID(id);
  n->setLevel(level);
  return n;
}
//-----------------------------------------------------------------------------
Node* GridData::createNode(int level, real x, real y, real z)
{
  // If a node exists with coordinates (x,y,z) then return a pointer to that 
  // node, else create a new node and return a pointer to that node.   
  Point pnt;
  for (List<Node>::Iterator n(&nodes); !n.end(); ++n){
    pnt = (n.pointer())->coord();
    if ( fabs(pnt.x-x)<DOLFIN_EPS ){
      if ( fabs(pnt.y-y)<DOLFIN_EPS ){
	if ( fabs(pnt.z-z)<DOLFIN_EPS ){
	  if ((n.pointer())->level() == level) return (n.pointer());
	}
      } 
    }
  }
  int id;
  Node *n = nodes.create(&id);
  n->setID(id);
  n->set(x,y,z);  
  n->setLevel(level);
  return n;
}
//-----------------------------------------------------------------------------
Cell* GridData::createCell(int level, Cell::Type type)
{
  int id;
  Cell *c = cells.create(&id);
  c->setID(id);
  c->init(type);
  c->setLevel(level);
  return c;
}
//-----------------------------------------------------------------------------
Cell* GridData::createCell(int level, Cell::Type type, int n0, int n1, int n2)
{
  int id;
  Cell *c = cells.create(&id);
  c->init(type);
  c->setID(id);
  c->set(getNode(n0),getNode(n1),getNode(n2));
  c->setLevel(level);
  return c;
}
//-----------------------------------------------------------------------------
Cell* GridData::createCell(int level, Cell::Type type, int n0, int n1, int n2, int n3)
{
  int id;
  Cell *c = cells.create(&id);
  c->init(type);
  c->setID(id);
  c->set(getNode(n0),getNode(n1),getNode(n2),getNode(n3));
  c->setLevel(level);
  return c;
}
//-----------------------------------------------------------------------------
Cell* GridData::createCell(int level, Cell::Type type, Node* n0, Node* n1, Node* n2)
{
  int id;
  Cell *c = cells.create(&id);
  c->init(type);
  c->setID(id);
  c->set(n0,n1,n2);
  c->setLevel(level);
  return c;
}
//-----------------------------------------------------------------------------
Cell* GridData::createCell(int level, Cell::Type type, Node* n0, Node* n1, Node* n2, Node* n3)
{
  int id;
  Cell *c = cells.create(&id);
  c->init(type);
  c->setID(id);
  c->set(n0,n1,n2,n3);
  c->setLevel(level);
  return c;
}
//-----------------------------------------------------------------------------
Edge* GridData::createEdge(int level)
{ 
  int id;
  Edge *e = edges.create(&id);
  e->setID(id);
  e->setLevel(level);
  return e;
}
//-----------------------------------------------------------------------------
Edge* GridData::createEdge(int level, int n0, int n1)
{
  // If an edge exists with nodes n0 and n1 then return a pointer to that 
  // edge, else create a new edge and return a pointer to that edge.   
  int en0,en1;
  for (List<Edge>::Iterator e(&edges); !e.end(); ++e){
    en0 = (e.pointer())->node(0)->id();
    en1 = (e.pointer())->node(1)->id();
    if ( n0 == en0 ){
      if ( (n1 == en1) && ((e.pointer())->level() == level) ) return (e.pointer());
    }
    if ( n0 == en1 ){
      if ( (n1 == en0) && ((e.pointer())->level() == level) ) return (e.pointer());
    }
  }
  int id;
  Edge *e = edges.create(&id);
  e->setID(id);
  e->set(getNode(n0),getNode(n1));
  e->setLevel(level);
  return e;
}
//-----------------------------------------------------------------------------
Edge* GridData::createEdge(int level, Node* n0, Node* n1)
{
  // If an edge exists with nodes n0 and n1 then return a pointer to that 
  // edge, else create a new edge and return a pointer to that edge.   
  int en0,en1;
  for (List<Edge>::Iterator e(&edges); !e.end(); ++e){
    en0 = (e.pointer())->node(0)->id();
    en1 = (e.pointer())->node(1)->id();
    if ( n0->id() == en0 ){
      if ( (n1->id() == en1) && ((e.pointer())->level() == level) ) return (e.pointer());
    }
    if ( n0->id() == en1 ){
      if ( (n1->id() == en0) && ((e.pointer())->level() == level) ) return (e.pointer());
    }
  }
  int id;
  Edge *e = edges.create(&id);
  e->setID(id);
  e->set(n0,n1);
  e->setLevel(level);
  return e;
}
//-----------------------------------------------------------------------------
Node* GridData::getNode(int id)
{
  return nodes.pointer(id);
}
//-----------------------------------------------------------------------------
Cell* GridData::getCell(int id)
{
  return cells.pointer(id);
}
//-----------------------------------------------------------------------------
Edge* GridData::getEdge(int id)
{
  return edges.pointer(id);
}
//-----------------------------------------------------------------------------
int GridData::noNodes() const
{
  return nodes.size();
}
//-----------------------------------------------------------------------------
int GridData::noCells() const
{
  return cells.size();
}
//-----------------------------------------------------------------------------
int GridData::noEdges() const
{
  return edges.size();
}
//-----------------------------------------------------------------------------
