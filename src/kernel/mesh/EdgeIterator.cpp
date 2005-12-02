// Copyright (C) 2003-2005 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// First added:  2003
// Last changed: 2005-12-01

#include <dolfin/Mesh.h>
#include <dolfin/Boundary.h>
#include <dolfin/Edge.h>
#include <dolfin/EdgeIterator.h>
#include <dolfin/GenericCell.h>
#include <dolfin/MeshData.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
// EdgeIterator
//-----------------------------------------------------------------------------
EdgeIterator::EdgeIterator(const Mesh &mesh)
{
  e = new MeshEdgeIterator(mesh);
}
//-----------------------------------------------------------------------------
EdgeIterator::EdgeIterator(const Mesh *mesh)
{
  e = new MeshEdgeIterator(*mesh);
}
//-----------------------------------------------------------------------------
EdgeIterator::EdgeIterator(const Boundary &boundary)
{
  e = new BoundaryEdgeIterator(boundary);
}
//-----------------------------------------------------------------------------
EdgeIterator::EdgeIterator(const Boundary *boundary)
{
  e = new BoundaryEdgeIterator(*boundary);
}
//-----------------------------------------------------------------------------
EdgeIterator::EdgeIterator(const Vertex& vertex)
{
  e = new VertexEdgeIterator(vertex);
}
//-----------------------------------------------------------------------------
EdgeIterator::EdgeIterator(const VertexIterator &vertexIterator)
{
  e = new VertexEdgeIterator(*vertexIterator);
}
//-----------------------------------------------------------------------------
EdgeIterator::EdgeIterator(const Cell &cell)
{
  e = new CellEdgeIterator(cell);
}
//-----------------------------------------------------------------------------
EdgeIterator::EdgeIterator(const CellIterator &cellIterator)
{
  e = new CellEdgeIterator(*cellIterator);
}
//-----------------------------------------------------------------------------
EdgeIterator::EdgeIterator(const Face& face)
{
  e = new FaceEdgeIterator(face);
}
//-----------------------------------------------------------------------------
EdgeIterator::EdgeIterator(const FaceIterator& faceIterator)
{
  e = new FaceEdgeIterator(*faceIterator);
}
//-----------------------------------------------------------------------------
EdgeIterator::operator EdgePointer() const
{
  return e->pointer();
}
//-----------------------------------------------------------------------------
EdgeIterator::~EdgeIterator()
{
  if ( e )
    delete e;
  e = 0;
}
//-----------------------------------------------------------------------------
EdgeIterator& EdgeIterator::operator++()
{
  ++(*e);

  return *this;
}
//-----------------------------------------------------------------------------
bool EdgeIterator::end()
{
  return e->end();
}
//-----------------------------------------------------------------------------
bool EdgeIterator::last()
{
  return e->last();
}
//-----------------------------------------------------------------------------
int EdgeIterator::index()
{
  return e->index();
}
//-----------------------------------------------------------------------------
Edge& EdgeIterator::operator*() const
{
  return *(*e);
}
//-----------------------------------------------------------------------------
Edge* EdgeIterator::operator->() const
{
  return e->pointer();
}
//-----------------------------------------------------------------------------
bool EdgeIterator::operator==(const EdgeIterator& e) const
{
  return this->e->pointer() == e.e->pointer();
}
//-----------------------------------------------------------------------------
bool EdgeIterator::operator!=(const EdgeIterator& e) const
{
  return this->e->pointer() != e.e->pointer();
}
//-----------------------------------------------------------------------------
// EdgeIterator::MeshEdgeIterator
//-----------------------------------------------------------------------------
EdgeIterator::MeshEdgeIterator::MeshEdgeIterator(const Mesh& mesh)
{
  edge_iterator = mesh.md->edges.begin();
  at_end = mesh.md->edges.end();
}
//-----------------------------------------------------------------------------
void EdgeIterator::MeshEdgeIterator::operator++()
{
  ++edge_iterator;
}
//-----------------------------------------------------------------------------
bool EdgeIterator::MeshEdgeIterator::end()
{
  return edge_iterator == at_end;
}
//-----------------------------------------------------------------------------
bool EdgeIterator::MeshEdgeIterator::last()
{
  return edge_iterator.last();
}
//-----------------------------------------------------------------------------
int EdgeIterator::MeshEdgeIterator::index()
{
  return edge_iterator.index();
}
//-----------------------------------------------------------------------------
Edge& EdgeIterator::MeshEdgeIterator::operator*() const
{
  return *edge_iterator;
}
//-----------------------------------------------------------------------------
Edge* EdgeIterator::MeshEdgeIterator::operator->() const
{
  return edge_iterator.pointer();
}
//-----------------------------------------------------------------------------
Edge* EdgeIterator::MeshEdgeIterator::pointer() const
{
  return edge_iterator.pointer();
}
//-----------------------------------------------------------------------------
// EdgeIterator::BoundaryEdgeIterator
//-----------------------------------------------------------------------------
EdgeIterator::BoundaryEdgeIterator::BoundaryEdgeIterator
(const Boundary& boundary) : edge_iterator(boundary.mesh->bd->edges)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
void EdgeIterator::BoundaryEdgeIterator::operator++()
{
  ++edge_iterator;
}
//-----------------------------------------------------------------------------
bool EdgeIterator::BoundaryEdgeIterator::end()
{
  return edge_iterator.end();
}
//-----------------------------------------------------------------------------
bool EdgeIterator::BoundaryEdgeIterator::last()
{
  return edge_iterator.last();
}
//-----------------------------------------------------------------------------
int EdgeIterator::BoundaryEdgeIterator::index()
{
  return edge_iterator.index();
}
//-----------------------------------------------------------------------------
Edge& EdgeIterator::BoundaryEdgeIterator::operator*() const
{
  return **edge_iterator;
}
//-----------------------------------------------------------------------------
Edge* EdgeIterator::BoundaryEdgeIterator::operator->() const
{
  return *edge_iterator;
}
//-----------------------------------------------------------------------------
Edge* EdgeIterator::BoundaryEdgeIterator::pointer() const
{
  return *edge_iterator;
}
//-----------------------------------------------------------------------------
// EdgeIterator::VertexEdgeIterator
//-----------------------------------------------------------------------------
EdgeIterator::VertexEdgeIterator::VertexEdgeIterator(const Vertex &vertex)
{
  edge_iterator = vertex.ne.begin();
}
//-----------------------------------------------------------------------------
void EdgeIterator::VertexEdgeIterator::operator++()
{
  ++edge_iterator;
}
//-----------------------------------------------------------------------------
bool EdgeIterator::VertexEdgeIterator::end()
{
  return edge_iterator.end();
}
//-----------------------------------------------------------------------------
bool EdgeIterator::VertexEdgeIterator::last()
{
  return edge_iterator.last();
}
//-----------------------------------------------------------------------------
int EdgeIterator::VertexEdgeIterator::index()
{
  return edge_iterator.index();
}
//-----------------------------------------------------------------------------
Edge& EdgeIterator::VertexEdgeIterator::operator*() const
{
  return **edge_iterator;
}
//-----------------------------------------------------------------------------
Edge* EdgeIterator::VertexEdgeIterator::operator->() const
{
  return *edge_iterator;
}
//-----------------------------------------------------------------------------
Edge* EdgeIterator::VertexEdgeIterator::pointer() const
{
  return *edge_iterator;
}
//-----------------------------------------------------------------------------
// EdgeIterator::CellEdgeIterator
//-----------------------------------------------------------------------------
EdgeIterator::CellEdgeIterator::CellEdgeIterator(const Cell &cell)
{
  edge_iterator = cell.c->ce.begin();
}
//-----------------------------------------------------------------------------
void EdgeIterator::CellEdgeIterator::operator++()
{
  ++edge_iterator;
}
//-----------------------------------------------------------------------------
bool EdgeIterator::CellEdgeIterator::end()
{
  return edge_iterator.end();
}
//-----------------------------------------------------------------------------
bool EdgeIterator::CellEdgeIterator::last()
{
  return edge_iterator.last();
}
//-----------------------------------------------------------------------------
int EdgeIterator::CellEdgeIterator::index()
{
  return edge_iterator.index();
}
//-----------------------------------------------------------------------------
Edge& EdgeIterator::CellEdgeIterator::operator*() const
{
  return **edge_iterator;
}
//-----------------------------------------------------------------------------
Edge* EdgeIterator::CellEdgeIterator::operator->() const
{
  return *edge_iterator;
}
//-----------------------------------------------------------------------------
Edge* EdgeIterator::CellEdgeIterator::pointer() const
{
  return *edge_iterator;
}
//-----------------------------------------------------------------------------
// EdgeIterator::FaceEdgeIterator
//-----------------------------------------------------------------------------
EdgeIterator::FaceEdgeIterator::FaceEdgeIterator(const Face &face)
{
  edge_iterator = face.fe.begin();
}
//-----------------------------------------------------------------------------
void EdgeIterator::FaceEdgeIterator::operator++()
{
  ++edge_iterator;
}
//-----------------------------------------------------------------------------
bool EdgeIterator::FaceEdgeIterator::end()
{
  return edge_iterator.end();
}
//-----------------------------------------------------------------------------
bool EdgeIterator::FaceEdgeIterator::last()
{
  return edge_iterator.last();
}
//-----------------------------------------------------------------------------
int EdgeIterator::FaceEdgeIterator::index()
{
  return edge_iterator.index();
}
//-----------------------------------------------------------------------------
Edge& EdgeIterator::FaceEdgeIterator::operator*() const
{
  return **edge_iterator;
}
//-----------------------------------------------------------------------------
Edge* EdgeIterator::FaceEdgeIterator::operator->() const
{
  return *edge_iterator;
}
//-----------------------------------------------------------------------------
Edge* EdgeIterator::FaceEdgeIterator::pointer() const
{
  return *edge_iterator;
}
//-----------------------------------------------------------------------------
