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
GridData::GridData(Grid& grid)
{
  this->grid = &grid;
}
//-----------------------------------------------------------------------------
GridData::~GridData()
{
  clear();
}
//-----------------------------------------------------------------------------
void GridData::clear()
{
  nodes.clear();
  cells.clear();
  edges.clear();
  faces.clear();
}
//-----------------------------------------------------------------------------
Node& GridData::createNode(Point p)
{
  return createNode(p.x, p.y, p.z);
}
//-----------------------------------------------------------------------------
Node& GridData::createNode(real x, real y, real z)
{
  int id;
  Node& n = nodes.create(id);
  n.set(x,y,z);  
  n.setID(id, *grid);
  return n;
}
//-----------------------------------------------------------------------------
Cell& GridData::createCell(int n0, int n1, int n2)
{
  int id;
  Cell& c = cells.create(id);
  c.set(getNode(n0), getNode(n1), getNode(n2));
  c.setID(id, *grid);
  return c;
}
//-----------------------------------------------------------------------------
Cell& GridData::createCell(int n0, int n1, int n2, int n3)
{
  int id;
  Cell& c = cells.create(id);
  c.set(getNode(n0), getNode(n1), getNode(n2), getNode(n3));
  c.setID(id, *grid);
  return c;
}
//-----------------------------------------------------------------------------
Cell& GridData::createCell(Node& n0, Node& n1, Node& n2)
{
  int id;
  Cell& c = cells.create(id);
  c.set(n0, n1, n2);
  c.setID(id, *grid);
  return c;
}
//-----------------------------------------------------------------------------
Cell& GridData::createCell(Node& n0, Node& n1, Node& n2, Node& n3)
{
  int id;
  Cell& c = cells.create(id);
  c.set(n0, n1, n2, n3);
  c.setID(id, *grid);
  return c;
}
//-----------------------------------------------------------------------------
Edge& GridData::createEdge(int n0, int n1)
{
  int id;
  Edge& e = edges.create(id);
  e.set(getNode(n0), getNode(n1));
  e.setID(id, *grid);
  return e;
}
//-----------------------------------------------------------------------------
Edge& GridData::createEdge(Node& n0, Node& n1)
{
  int id;
  Edge& e = edges.create(id);
  e.set(n0, n1);
  e.setID(id, *grid);
  return e;
}
//-----------------------------------------------------------------------------
Face& GridData::createFace(int e0, int e1, int e2)
{
  int id;
  Face& f = faces.create(id);
  f.set(getEdge(e0), getEdge(e1), getEdge(e2));
  f.setID(id, *grid);
  return f;
}
//-----------------------------------------------------------------------------
Face& GridData::createFace(Edge& e0, Edge& e1, Edge& e2)
{
  int id;
  Face& f = faces.create(id);
  f.set(e0, e1, e2);
  f.setID(id, *grid);
  return f;
}
//-----------------------------------------------------------------------------
Node& GridData::getNode(int id)
{
  return nodes(id);
}
//-----------------------------------------------------------------------------
Cell& GridData::getCell(int id)
{
  return cells(id);
}
//-----------------------------------------------------------------------------
Edge& GridData::getEdge(int id)
{
  return edges(id);
}
//-----------------------------------------------------------------------------
Face& GridData::getFace(int id)
{
  return faces(id);
}
//-----------------------------------------------------------------------------
void GridData::remove(Node& node)
{
  nodes.remove(node);
}
//-----------------------------------------------------------------------------
void GridData::remove(Cell& cell)
{
  cells.remove(cell);
}
//-----------------------------------------------------------------------------
void GridData::remove(Edge& edge)
{
  edges.remove(edge);
}
//-----------------------------------------------------------------------------
void GridData::remove(Face& face)
{
  faces.remove(face);
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
int GridData::noFaces() const
{
  return faces.size();
}
//-----------------------------------------------------------------------------
void GridData::setGrid(Grid& grid)
{
  // Change the grid pointer in all data
  for (Table<Node>::Iterator n(nodes); !n.end(); ++n)
    n->setGrid(grid);

  for (Table<Cell>::Iterator c(cells); !c.end(); ++c)
    c->setGrid(grid);

  for (Table<Edge>::Iterator e(edges); !e.end(); ++e)
    e->setGrid(grid);

  for (Table<Face>::Iterator f(faces); !f.end(); ++f)
    f->setGrid(grid);

  // Change grid pointer
  this->grid = &grid;
}
//-----------------------------------------------------------------------------
