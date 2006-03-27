// Copyright (C) 2003-2006 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// First added:  2003
// Last changed: 2006-03-24

#include <dolfin/dolfin_log.h>
#include <dolfin/Vertex.h>
#include <dolfin/Edge.h>
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
Cell::Cell(Vertex& n0, Vertex& n1, Vertex& n2)
{
  c = new Triangle(n0, n1, n2);
}
//-----------------------------------------------------------------------------
Cell::Cell(Vertex& n0, Vertex& n1, Vertex& n2, Vertex& n3)
{
  c = new Tetrahedron(n0, n1, n2, n3);
}
//-----------------------------------------------------------------------------
Cell::~Cell()
{
  clear();
}
//-----------------------------------------------------------------------------
void Cell::clear()
{
  if ( c )
    delete c;
  c = 0;
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
Cell::Orientation Cell::orientation() const
{
  dolfin_assert(c);
  return c->orientation();
}
//-----------------------------------------------------------------------------
int Cell::numVertices() const
{
  if ( c )
    return c->numVertices();

  return 0;
}
//-----------------------------------------------------------------------------
int Cell::numEdges() const
{
  if ( c )
    return c->numEdges();

  return 0;
}
//-----------------------------------------------------------------------------
int Cell::numFaces() const
{
  if ( c )
    return c->numFaces();

  return 0;
}
//-----------------------------------------------------------------------------
int Cell::numBoundaries() const
{
  if ( c )
    return c->numBoundaries();

  return 0;
}
//-----------------------------------------------------------------------------
int Cell::numCellNeighbors() const
{
  if ( c )
    return c->numCellNeighbors();

  return 0;
}
//-----------------------------------------------------------------------------
int Cell::numVertexNeighbors() const
{
  if ( c )
    return c->numVertexNeighbors();

  return 0;
}
//-----------------------------------------------------------------------------
int Cell::numChildren() const
{
  if ( c )
    return c->numChildren();

  return 0;
}
//-----------------------------------------------------------------------------
Vertex& Cell::vertex(int i) const
{
  dolfin_assert(c);
  return c->vertex(i);
}
//-----------------------------------------------------------------------------
Edge& Cell::edge(int i) const
{
  dolfin_assert(c);
  return c->edge(i);
}
//-----------------------------------------------------------------------------
Face& Cell::face(int i) const
{
  dolfin_assert(c);
  return c->face(i);
}
//-----------------------------------------------------------------------------
Cell& Cell::neighbor(int i) const
{
  dolfin_assert(c);
  return c->neighbor(i);
}
//-----------------------------------------------------------------------------
Cell* Cell::parent() const
{
  dolfin_assert(c);
  return c->parent();
}
//-----------------------------------------------------------------------------
Cell* Cell::child(int i) const
{
  dolfin_assert(c);
  return c->child(i);
}
//-----------------------------------------------------------------------------
Mesh& Cell::mesh()
{
  dolfin_assert(c);
  return *(c->_mesh);
}
//-----------------------------------------------------------------------------
const Mesh& Cell::mesh() const
{
  dolfin_assert(c);
  return *(c->_mesh);
}
//-----------------------------------------------------------------------------
Point& Cell::coord(int i) const
{ 
  dolfin_assert(c);
  return c->coord(i);
}
//-----------------------------------------------------------------------------
Point Cell::midpoint() const
{
  dolfin_assert(c);
  return c->midpoint();
}
//-----------------------------------------------------------------------------
int Cell::vertexID(int i) const
{
  dolfin_assert(c);
  return c->vertexID(i);
}
//-----------------------------------------------------------------------------
int Cell::edgeID(int i) const
{
  dolfin_assert(c);
  return c->edgeID(i);
}
//-----------------------------------------------------------------------------
int Cell::faceID(int i) const
{
  dolfin_assert(c);
  return c->faceID(i);
}
//-----------------------------------------------------------------------------
real Cell::volume() const
{
  dolfin_assert(c);
  return c->volume();
}
//-----------------------------------------------------------------------------
real Cell::diameter() const
{
  dolfin_assert(c);
  return c->diameter();
}
//-----------------------------------------------------------------------------
dolfin::uint Cell::edgeAlignment(uint i) const
{
  dolfin_assert(c);
  return c->edgeAlignment(i);
}
//-----------------------------------------------------------------------------
dolfin::uint Cell::faceAlignment(uint i) const
{
  dolfin_assert(c);
  return c->faceAlignment(i);
}
//-----------------------------------------------------------------------------
bool Cell::operator==(const Cell& cell) const
{
  return this == &cell;
}
//-----------------------------------------------------------------------------
bool Cell::operator!=(const Cell& cell) const
{
  return this != &cell;
}
//-----------------------------------------------------------------------------
void Cell::mark(bool refine)
{
  dolfin_assert(c);
  c->mark(refine);
}
//-----------------------------------------------------------------------------
dolfin::LogStream& dolfin::operator<<(LogStream& stream, const Cell& cell)
{
  switch ( cell.type() ){
  case Cell::triangle:
    stream << "[ Cell (triangle): id = " << cell.id() << " vertices = ( ";
    for (VertexIterator n(cell); !n.end(); ++n)
      stream << n->id() << " ";
    stream << ") ]";
    break;
  case Cell::tetrahedron:
    stream << "[ Cell (tetrahedron): id = " << cell.id() << " vertices = ( ";
    for (VertexIterator n(cell); !n.end(); ++n)
      stream << n->id() << " ";
    stream << ") ]";
    break;
  default:
    dolfin_error("Unknown cell type");
  }	 
  
  return stream;
}
//-----------------------------------------------------------------------------
int Cell::setID(int id, Mesh& mesh)
{
  dolfin_assert(c);
  return c->setID(id, mesh);
}
//-----------------------------------------------------------------------------
void Cell::setMesh(Mesh& mesh)
{
  dolfin_assert(c);
  c->setMesh(mesh);
}
//-----------------------------------------------------------------------------
void Cell::setParent(Cell& parent)
{
  dolfin_assert(c);
  c->setParent(parent);
}
//-----------------------------------------------------------------------------
void Cell::removeParent()
{
  dolfin_assert(c);
  c->removeParent();
}
//-----------------------------------------------------------------------------
void Cell::initChildren(int n)
{
  dolfin_assert(c);
  c->initChildren(n);
}
//-----------------------------------------------------------------------------
void Cell::addChild(Cell& child)
{
  dolfin_assert(c);
  c->addChild(child);
}
//-----------------------------------------------------------------------------
void Cell::removeChild(Cell& child)
{
  dolfin_assert(c);
  c->removeChild(child);
}
//-----------------------------------------------------------------------------
void Cell::set(Vertex& n0, Vertex& n1, Vertex& n2)
{
  if ( c )
    delete c;
  
  c = new Triangle(n0, n1, n2);
}
//-----------------------------------------------------------------------------
void Cell::set(Vertex& n0, Vertex& n1, Vertex& n2, Vertex& n3)
{
  if ( c )
    delete c;

  c = new Tetrahedron(n0, n1, n2, n3);
}
//-----------------------------------------------------------------------------
bool Cell::neighbor(Cell& cell) const
{
  dolfin_assert(c);
  return c->neighbor(*cell.c);
}
//-----------------------------------------------------------------------------
bool Cell::haveVertex(Vertex& vertex) const
{
  dolfin_assert(c);
  return c->haveVertex(vertex);
}
//-----------------------------------------------------------------------------
bool Cell::haveEdge(Edge& edge) const
{
  dolfin_assert(c);
  return c->haveEdge(edge);
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
void Cell::createEdge(Vertex& n0, Vertex& n1)
{
  dolfin_assert(c);
  c->createEdge(n0, n1);
}
//-----------------------------------------------------------------------------
void Cell::createFace(Edge& e0, Edge& e1, Edge& e2)
{
  dolfin_assert(c);
  c->createFace(e0, e1, e2);
}
//-----------------------------------------------------------------------------
Vertex* Cell::findVertex(const Point& p) const
{
  dolfin_assert(c);
  return c->findVertex(p);
}
//-----------------------------------------------------------------------------
Edge* Cell::findEdge(Vertex& n0, Vertex& n1)
{
  dolfin_assert(c);
  return c->findEdge(n0, n1);
}
//-----------------------------------------------------------------------------
Face* Cell::findFace(Edge& e0, Edge& e1, Edge& e2)
{
  dolfin_assert(c);
  return c->findFace(e0, e1, e2);
}
//-----------------------------------------------------------------------------
Face* Cell::findFace(Edge& e0, Edge& e1)
{
  dolfin_assert(c);
  return c->findFace(e0, e1);
}
//-----------------------------------------------------------------------------
Cell::Marker& Cell::marker()
{
  dolfin_assert(c);
  return c->marker();
}
//-----------------------------------------------------------------------------
Cell::Status& Cell::status()
{
  dolfin_assert(c);
  return c->status();
}
//-----------------------------------------------------------------------------
void Cell::sort()
{
  dolfin_assert(c);
  c->sort();
}
//-----------------------------------------------------------------------------
