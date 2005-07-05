// Copyright (C) 2002-2005 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// First added:  2002
// Last changed: 2005

#include <cmath>
#include <dolfin/Node.h>
#include <dolfin/Triangle.h>
#include <dolfin/Tetrahedron.h>
#include <dolfin/Cell.h>
#include <dolfin/MeshData.h>
#include <dolfin/constants.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
MeshData::MeshData(Mesh& mesh)
{
  this->mesh = &mesh;
}
//-----------------------------------------------------------------------------
MeshData::~MeshData()
{
  clear();
}
//-----------------------------------------------------------------------------
void MeshData::clear()
{
  nodes.clear();
  cells.clear();
  edges.clear();
  faces.clear();
}
//-----------------------------------------------------------------------------
Node& MeshData::createNode(Point p)
{
  return createNode(p.x, p.y, p.z);
}
//-----------------------------------------------------------------------------
Node& MeshData::createNode(real x, real y, real z)
{
  int id;
  Node& n = nodes.create(id);
  n.set(x,y,z);  
  n.setID(id, *mesh);
  return n;
}
//-----------------------------------------------------------------------------
Cell& MeshData::createCell(int n0, int n1, int n2)
{
  int id;
  Cell& c = cells.create(id);
  c.set(node(n0), node(n1), node(n2));
  c.setID(id, *mesh);
  return c;
}
//-----------------------------------------------------------------------------
Cell& MeshData::createCell(int n0, int n1, int n2, int n3)
{
  int id;
  Cell& c = cells.create(id);
  c.set(node(n0), node(n1), node(n2), node(n3));
  c.setID(id, *mesh);
  return c;
}
//-----------------------------------------------------------------------------
Cell& MeshData::createCell(Node& n0, Node& n1, Node& n2)
{
  int id;
  Cell& c = cells.create(id);
  c.set(n0, n1, n2);
  c.setID(id, *mesh);
  return c;
}
//-----------------------------------------------------------------------------
Cell& MeshData::createCell(Node& n0, Node& n1, Node& n2, Node& n3)
{
  int id;
  Cell& c = cells.create(id);
  c.set(n0, n1, n2, n3);
  c.setID(id, *mesh);
  return c;
}
//-----------------------------------------------------------------------------
Edge& MeshData::createEdge(int n0, int n1)
{
  int id;
  Edge& e = edges.create(id);
  e.set(node(n0), node(n1));
  e.setID(id, *mesh);
  return e;
}
//-----------------------------------------------------------------------------
Edge& MeshData::createEdge(Node& n0, Node& n1)
{
  int id;
  Edge& e = edges.create(id);
  e.set(n0, n1);
  e.setID(id, *mesh);
  return e;
}
//-----------------------------------------------------------------------------
Face& MeshData::createFace(int e0, int e1, int e2)
{
  int id;
  Face& f = faces.create(id);
  f.set(edge(e0), edge(e1), edge(e2));
  f.setID(id, *mesh);
  return f;
}
//-----------------------------------------------------------------------------
Face& MeshData::createFace(Edge& e0, Edge& e1, Edge& e2)
{
  int id;
  Face& f = faces.create(id);
  f.set(e0, e1, e2);
  f.setID(id, *mesh);
  return f;
}
//-----------------------------------------------------------------------------
Node& MeshData::node(int id)
{
  return nodes(id);
}
//-----------------------------------------------------------------------------
Cell& MeshData::cell(int id)
{
  return cells(id);
}
//-----------------------------------------------------------------------------
Edge& MeshData::edge(int id)
{
  return edges(id);
}
//-----------------------------------------------------------------------------
Face& MeshData::face(int id)
{
  return faces(id);
}
//-----------------------------------------------------------------------------
void MeshData::remove(Node& node)
{
  node.clear();
  nodes.remove(node);
}
//-----------------------------------------------------------------------------
void MeshData::remove(Cell& cell)
{
  cell.clear();
  cells.remove(cell);
}
//-----------------------------------------------------------------------------
void MeshData::remove(Edge& edge)
{
  edge.clear();
  edges.remove(edge);
}
//-----------------------------------------------------------------------------
void MeshData::remove(Face& face)
{
  face.clear();
  faces.remove(face);
}
//-----------------------------------------------------------------------------
int MeshData::noNodes() const
{
  return nodes.size();
}
//-----------------------------------------------------------------------------
int MeshData::noCells() const
{
  return cells.size();
}
//-----------------------------------------------------------------------------
int MeshData::noEdges() const
{
  return edges.size();
}
//-----------------------------------------------------------------------------
int MeshData::noFaces() const
{
  return faces.size();
}
//-----------------------------------------------------------------------------
void MeshData::setMesh(Mesh& mesh)
{
  // Change the mesh pointer in all data
  for (Table<Node>::Iterator n(nodes); !n.end(); ++n)
    n->setMesh(mesh);

  for (Table<Cell>::Iterator c(cells); !c.end(); ++c)
    c->setMesh(mesh);

  for (Table<Edge>::Iterator e(edges); !e.end(); ++e)
    e->setMesh(mesh);

  for (Table<Face>::Iterator f(faces); !f.end(); ++f)
    f->setMesh(mesh);

  // Change mesh pointer
  this->mesh = &mesh;
}
//-----------------------------------------------------------------------------
