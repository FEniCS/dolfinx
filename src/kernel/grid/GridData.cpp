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
  if (level > _finest_grid_level) _finest_grid_level = level; 
  return n;
}
//-----------------------------------------------------------------------------
Node* GridData::createNode(int level, real x, real y, real z)
{
  // If a node exists with coordinates (x,y,z) then return a pointer to that 
  // node, else create a new node and return a pointer to that node.   
  Point pnt;
  for (List<Node>::Iterator n(nodes); !n.end(); ++n){
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
  if (level > _finest_grid_level) _finest_grid_level = level; 
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
  if (level > _finest_grid_level) _finest_grid_level = level; 
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
  if (level > _finest_grid_level) _finest_grid_level = level; 
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
  if (level > _finest_grid_level) _finest_grid_level = level; 
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
  if (level > _finest_grid_level) _finest_grid_level = level; 
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
  if (level > _finest_grid_level) _finest_grid_level = level; 
  return c;
}
//-----------------------------------------------------------------------------
void GridData::createEdges(Cell* c)
{
  // If an edge exists with nodes n0 and n1 then return a pointer to that 
  // edge, else create a new edge and return a pointer to that edge.   
  Edge *edge;
  int id;
  bool edge_exists;
  for (int i=0;i<c->noEdges();i++){ 
    edge_exists = false;
    for (List<Edge>::Iterator e(edges); !e.end(); ++e){
      switch(i){ 
      case 0: 
	if ( ( ((e.pointer())->node(0)->id() == c->node(0)->id()) && 
	       ((e.pointer())->node(1)->id() == c->node(1)->id()) && 
	       ((e.pointer())->level() == c->level()) ) ||
	     ( ((e.pointer())->node(1)->id() == c->node(0)->id()) && 
	       ((e.pointer())->node(0)->id() == c->node(1)->id()) && 
	       ((e.pointer())->level() == c->level()) ) ){
	  c->setEdge(e.pointer(),i);
	  edge_exists = true;
	  break;
	}
	break;
      case 1: 
	if ( ( ((e.pointer())->node(0)->id() == c->node(0)->id()) && 
	       ((e.pointer())->node(1)->id() == c->node(2)->id()) && 
	       ((e.pointer())->level() == c->level()) ) ||
	     ( ((e.pointer())->node(1)->id() == c->node(0)->id()) && 
	       ((e.pointer())->node(0)->id() == c->node(2)->id()) && 
	       ((e.pointer())->level() == c->level()) ) ){
	  c->setEdge(e.pointer(),i);
	  edge_exists = true;
	  break;
	}
	break;
      case 2: 
	if ( ( ((e.pointer())->node(0)->id() == c->node(0)->id()) && 
	       ((e.pointer())->node(1)->id() == c->node(3)->id()) && 
	       ((e.pointer())->level() == c->level()) ) ||
	     ( ((e.pointer())->node(1)->id() == c->node(0)->id()) && 
	       ((e.pointer())->node(0)->id() == c->node(3)->id()) && 
	       ((e.pointer())->level() == c->level()) ) ){
	  c->setEdge(e.pointer(),i);
	  edge_exists = true;
	  break;
	}
	break;
      case 3: 
	if ( ( ((e.pointer())->node(0)->id() == c->node(1)->id()) && 
	       ((e.pointer())->node(1)->id() == c->node(2)->id()) && 
	       ((e.pointer())->level() == c->level()) ) ||
	     ( ((e.pointer())->node(1)->id() == c->node(1)->id()) && 
	       ((e.pointer())->node(0)->id() == c->node(2)->id()) && 
	       ((e.pointer())->level() == c->level()) ) ){
	  c->setEdge(e.pointer(),i);
	  edge_exists = true;
	  break;
	}
	break;
      case 4: 
	if ( ( ((e.pointer())->node(0)->id() == c->node(1)->id()) && 
	       ((e.pointer())->node(1)->id() == c->node(3)->id()) && 
	       ((e.pointer())->level() == c->level()) ) ||
	     ( ((e.pointer())->node(1)->id() == c->node(1)->id()) && 
	       ((e.pointer())->node(0)->id() == c->node(3)->id()) && 
	       ((e.pointer())->level() == c->level()) ) ){
	  c->setEdge(e.pointer(),i);
	  edge_exists = true;
	  break;
	}
	break;
      case 5: 
	if ( ( ((e.pointer())->node(0)->id() == c->node(2)->id()) && 
	       ((e.pointer())->node(1)->id() == c->node(3)->id()) && 
	       ((e.pointer())->level() == c->level()) ) ||
	     ( ((e.pointer())->node(1)->id() == c->node(2)->id()) && 
	       ((e.pointer())->node(0)->id() == c->node(3)->id()) && 
	       ((e.pointer())->level() == c->level()) ) ){
	  c->setEdge(e.pointer(),i);
	  edge_exists = true;
	  break;
	}
	break;
      default:
	dolfin_error("wrong counter value on i");
      }
    }
    
    if (!edge_exists){
      edge = edges.create(&id);
      edge->setID(id);
      switch(i){ 
      case 0: 
	edge->set(c->node(0),c->node(1));
	break;
      case 1: 
	edge->set(c->node(0),c->node(2));
	break;
      case 2: 
	edge->set(c->node(0),c->node(3));
	break;
      case 3: 
	edge->set(c->node(1),c->node(2));
	break;
      case 4: 
	edge->set(c->node(1),c->node(3));
	break;
      case 5: 
	edge->set(c->node(2),c->node(3));
	break;
      default:
	dolfin_error("wrong counter value on i");
      }
      edge->setLevel(c->level());
      if (c->level() > _finest_grid_level) _finest_grid_level = c->level(); 
      c->setEdge(edge,i);
    }    
  }
}
//-----------------------------------------------------------------------------
void GridData::setFinestGridLevel(int gl)
{
  _finest_grid_level = gl;
}
//-----------------------------------------------------------------------------
int GridData::finestGridLevel()
{
  return _finest_grid_level;
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
