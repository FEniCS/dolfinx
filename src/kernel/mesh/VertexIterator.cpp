// Copyright (C) 2002-2005 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// First added:  2002
// Last changed: 2005-12-01

#include <dolfin/Mesh.h>
#include <dolfin/Boundary.h>
#include <dolfin/Vertex.h>
#include <dolfin/VertexIterator.h>
#include <dolfin/GenericCell.h>
#include <dolfin/MeshData.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
// VertexIterator
//-----------------------------------------------------------------------------
VertexIterator::VertexIterator(const Mesh &mesh)
{
  n = new MeshVertexIterator(mesh);
}
//-----------------------------------------------------------------------------
VertexIterator::VertexIterator(const Mesh *mesh)
{
  n = new MeshVertexIterator(*mesh);
}
//-----------------------------------------------------------------------------
VertexIterator::VertexIterator(const Boundary& boundary)
{
  n = new BoundaryVertexIterator(boundary);
}
//-----------------------------------------------------------------------------
VertexIterator::VertexIterator(const Vertex &vertex)
{
  n = new VertexVertexIterator(vertex);
}
//-----------------------------------------------------------------------------
VertexIterator::VertexIterator(const VertexIterator &vertexIterator)
{
  n = new VertexVertexIterator(*vertexIterator);
}
//-----------------------------------------------------------------------------
VertexIterator::VertexIterator(const Cell &cell)
{
  n = new CellVertexIterator(cell);
}
//-----------------------------------------------------------------------------
VertexIterator::VertexIterator(const CellIterator &cellIterator)
{
  n = new CellVertexIterator(*cellIterator);
}
//-----------------------------------------------------------------------------
VertexIterator::operator VertexPointer() const
{
  return n->pointer();
}
//-----------------------------------------------------------------------------
VertexIterator::~VertexIterator()
{
  delete n;
}
//-----------------------------------------------------------------------------
VertexIterator& VertexIterator::operator++()
{
  ++(*n);

  return *this;
}
//-----------------------------------------------------------------------------
bool VertexIterator::end()
{
  return n->end();
}
//-----------------------------------------------------------------------------
bool VertexIterator::last()
{
  return n->last();
}
//-----------------------------------------------------------------------------
int VertexIterator::index()
{
  return n->index();
}
//-----------------------------------------------------------------------------
Vertex& VertexIterator::operator*() const
{
  return *(*n);
}
//-----------------------------------------------------------------------------
Vertex* VertexIterator::operator->() const
{
  return n->pointer();
}
//-----------------------------------------------------------------------------
bool VertexIterator::operator==(const VertexIterator& n) const
{
  return this->n->pointer() == n.n->pointer();
}
//-----------------------------------------------------------------------------
bool VertexIterator::operator!=(const VertexIterator& n) const
{
  return this->n->pointer() != n.n->pointer();
}
//-----------------------------------------------------------------------------
bool VertexIterator::operator==(const Vertex& n) const
{
  return this->n->pointer() == &n;
}
//-----------------------------------------------------------------------------
bool VertexIterator::operator!=(const Vertex& n) const
{
  return this->n->pointer() != &n;
}
//-----------------------------------------------------------------------------
// VertexIterator::MeshVertexIterator
//-----------------------------------------------------------------------------
VertexIterator::MeshVertexIterator::MeshVertexIterator(const Mesh& mesh)
{
  vertex_iterator = mesh.md->vertices.begin();
  at_end = mesh.md->vertices.end();
}
//-----------------------------------------------------------------------------
void VertexIterator::MeshVertexIterator::operator++()
{
  ++vertex_iterator;
}
//-----------------------------------------------------------------------------
bool VertexIterator::MeshVertexIterator::end()
{
  return vertex_iterator == at_end;
}
//-----------------------------------------------------------------------------
bool VertexIterator::MeshVertexIterator::last()
{
  return vertex_iterator.last();
}
//-----------------------------------------------------------------------------
int VertexIterator::MeshVertexIterator::index()
{
  return vertex_iterator.index();
}
//-----------------------------------------------------------------------------
Vertex& VertexIterator::MeshVertexIterator::operator*() const
{
  return *vertex_iterator;
}
//-----------------------------------------------------------------------------
Vertex* VertexIterator::MeshVertexIterator::operator->() const
{
  return vertex_iterator.pointer();
}
//-----------------------------------------------------------------------------
Vertex* VertexIterator::MeshVertexIterator::pointer() const
{
  return vertex_iterator.pointer();
}
//-----------------------------------------------------------------------------
// VertexIterator::BoundaryVertexIterator
//-----------------------------------------------------------------------------
VertexIterator::BoundaryVertexIterator::BoundaryVertexIterator
(const Boundary& boundary) : vertex_iterator(boundary.mesh->bd->vertices)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
void VertexIterator::BoundaryVertexIterator::operator++()
{
  ++vertex_iterator;
}
//-----------------------------------------------------------------------------
bool VertexIterator::BoundaryVertexIterator::end()
{
  return vertex_iterator.end();
}
//-----------------------------------------------------------------------------
bool VertexIterator::BoundaryVertexIterator::last()
{
  return vertex_iterator.last();
}
//-----------------------------------------------------------------------------
int VertexIterator::BoundaryVertexIterator::index()
{
  return vertex_iterator.index();
}
//-----------------------------------------------------------------------------
Vertex& VertexIterator::BoundaryVertexIterator::operator*() const
{
  return **vertex_iterator;
}
//-----------------------------------------------------------------------------
Vertex* VertexIterator::BoundaryVertexIterator::operator->() const
{
  return *vertex_iterator;
}
//-----------------------------------------------------------------------------
Vertex* VertexIterator::BoundaryVertexIterator::pointer() const
{
  return *vertex_iterator;
}
//-----------------------------------------------------------------------------
// VertexIterator::CellVertexIterator
//-----------------------------------------------------------------------------
VertexIterator::CellVertexIterator::CellVertexIterator(const Cell &cell)
{
  vertex_iterator = cell.c->cn.begin();
}
//-----------------------------------------------------------------------------
void VertexIterator::CellVertexIterator::operator++()
{
  ++vertex_iterator;
}
//-----------------------------------------------------------------------------
bool VertexIterator::CellVertexIterator::end()
{
  return vertex_iterator.end();
}
//-----------------------------------------------------------------------------
bool VertexIterator::CellVertexIterator::last()
{
  return vertex_iterator.last();
}
//-----------------------------------------------------------------------------
int VertexIterator::CellVertexIterator::index()
{
  return vertex_iterator.index();
}
//-----------------------------------------------------------------------------
Vertex& VertexIterator::CellVertexIterator::operator*() const
{
  return **vertex_iterator;
}
//-----------------------------------------------------------------------------
Vertex* VertexIterator::CellVertexIterator::operator->() const
{
  return *vertex_iterator;
}
//-----------------------------------------------------------------------------
Vertex* VertexIterator::CellVertexIterator::pointer() const
{
  return *vertex_iterator;
}
//-----------------------------------------------------------------------------
// VertexIterator::VertexVertexIterator
//-----------------------------------------------------------------------------
VertexIterator::VertexVertexIterator::VertexVertexIterator(const Vertex &vertex)
{
  vertex_iterator = vertex.nn.begin();
}
//-----------------------------------------------------------------------------
void VertexIterator::VertexVertexIterator::operator++()
{
  ++vertex_iterator;
}
//-----------------------------------------------------------------------------
bool VertexIterator::VertexVertexIterator::end()
{
  return vertex_iterator.end();
}
//-----------------------------------------------------------------------------
bool VertexIterator::VertexVertexIterator::last()
{
  return vertex_iterator.last();
}
//-----------------------------------------------------------------------------
int VertexIterator::VertexVertexIterator::index()
{
  return vertex_iterator.index();
}
//-----------------------------------------------------------------------------
Vertex& VertexIterator::VertexVertexIterator::operator*() const
{
  return **vertex_iterator;
}
//-----------------------------------------------------------------------------
Vertex* VertexIterator::VertexVertexIterator::operator->() const
{
  return *vertex_iterator;
}
//-----------------------------------------------------------------------------
Vertex* VertexIterator::VertexVertexIterator::pointer() const
{
  return *vertex_iterator;
}
//-----------------------------------------------------------------------------
