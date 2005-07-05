// Copyright (C) 2003-2005 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// First added:  2003
// Last changed: 2005

#include <dolfin/Mesh.h>
#include <dolfin/Boundary.h>
#include <dolfin/Cell.h>
#include <dolfin/GenericCell.h>
#include <dolfin/FaceIterator.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
// FaceIterator
//-----------------------------------------------------------------------------
FaceIterator::FaceIterator(const Mesh &mesh)
{
  f = new MeshFaceIterator(mesh);
}
//-----------------------------------------------------------------------------
FaceIterator::FaceIterator(const Mesh *mesh)
{
  f = new MeshFaceIterator(*mesh);
}
//-----------------------------------------------------------------------------
FaceIterator::FaceIterator(const Boundary &boundary)
{
  f = new BoundaryFaceIterator(boundary);
}
//-----------------------------------------------------------------------------
FaceIterator::FaceIterator(const Boundary *boundary)
{
  f = new BoundaryFaceIterator(*boundary);
}
//-----------------------------------------------------------------------------
FaceIterator::FaceIterator(const Cell& cell)
{
  f = new CellFaceIterator(cell);
}
//-----------------------------------------------------------------------------
FaceIterator::FaceIterator(const CellIterator& cellIterator)
{
  f = new CellFaceIterator(*cellIterator);
}
//-----------------------------------------------------------------------------
FaceIterator::operator FacePointer() const
{
  return f->pointer();
}
//-----------------------------------------------------------------------------
FaceIterator::~FaceIterator()
{
  delete f;
}
//-----------------------------------------------------------------------------
FaceIterator& FaceIterator::operator++()
{
  ++(*f);

  return *this;
}
//-----------------------------------------------------------------------------
bool FaceIterator::end()
{
  return f->end();
}
//-----------------------------------------------------------------------------
bool FaceIterator::last()
{
  return f->last();
}
//-----------------------------------------------------------------------------
int FaceIterator::index()
{
  return f->index();
}
//-----------------------------------------------------------------------------
Face& FaceIterator::operator*() const
{
  return *(*f);
}
//-----------------------------------------------------------------------------
Face* FaceIterator::operator->() const
{
  return f->pointer();
}
//-----------------------------------------------------------------------------
bool FaceIterator::operator==(const FaceIterator& f) const
{
  return this->f->pointer() == f.f->pointer();
}
//-----------------------------------------------------------------------------
bool FaceIterator::operator!=(const FaceIterator& f) const
{
  return this->f->pointer() != f.f->pointer();
}
//-----------------------------------------------------------------------------
bool FaceIterator::operator==(const Face& f) const
{
  return this->f->pointer() == &f;
}
//-----------------------------------------------------------------------------
bool FaceIterator::operator!=(const Face& f) const
{
  return this->f->pointer() == &f;
}
//-----------------------------------------------------------------------------
// FaceIterator::MeshFaceIterator
//-----------------------------------------------------------------------------
FaceIterator::MeshFaceIterator::MeshFaceIterator(const Mesh& mesh)
{
  face_iterator = mesh.md->faces.begin();
  at_end = mesh.md->faces.end();
}
//-----------------------------------------------------------------------------
void FaceIterator::MeshFaceIterator::operator++()
{
  ++face_iterator;
}
//-----------------------------------------------------------------------------
bool FaceIterator::MeshFaceIterator::end()
{
  return face_iterator == at_end;
}
//-----------------------------------------------------------------------------
bool FaceIterator::MeshFaceIterator::last()
{
  return face_iterator.last();
}
//-----------------------------------------------------------------------------
int FaceIterator::MeshFaceIterator::index()
{
  return face_iterator.index();
}
//-----------------------------------------------------------------------------
Face& FaceIterator::MeshFaceIterator::operator*() const
{
  return *face_iterator;
}
//-----------------------------------------------------------------------------
Face* FaceIterator::MeshFaceIterator::operator->() const
{
  return face_iterator.pointer();
}
//-----------------------------------------------------------------------------
Face* FaceIterator::MeshFaceIterator::pointer() const
{
  return face_iterator.pointer();
}
//-----------------------------------------------------------------------------
// FaceIterator::BoundaryFaceIterator
//-----------------------------------------------------------------------------
FaceIterator::BoundaryFaceIterator::BoundaryFaceIterator
(const Boundary& boundary) : face_iterator(boundary.mesh->bd->faces)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
void FaceIterator::BoundaryFaceIterator::operator++()
{
  ++face_iterator;
}
//-----------------------------------------------------------------------------
bool FaceIterator::BoundaryFaceIterator::end()
{
  return face_iterator.end();
}
//-----------------------------------------------------------------------------
bool FaceIterator::BoundaryFaceIterator::last()
{
  return face_iterator.last();
}
//-----------------------------------------------------------------------------
int FaceIterator::BoundaryFaceIterator::index()
{
  return face_iterator.index();
}
//-----------------------------------------------------------------------------
Face& FaceIterator::BoundaryFaceIterator::operator*() const
{
  return **face_iterator;
}
//-----------------------------------------------------------------------------
Face* FaceIterator::BoundaryFaceIterator::operator->() const
{
  return *face_iterator;
}
//-----------------------------------------------------------------------------
Face* FaceIterator::BoundaryFaceIterator::pointer() const
{
  return *face_iterator;
}
//-----------------------------------------------------------------------------
// FaceIterator::CellFaceIterator
//-----------------------------------------------------------------------------
FaceIterator::CellFaceIterator::CellFaceIterator(const Cell &cell)
{
  face_iterator = cell.c->cf.begin();
}
//-----------------------------------------------------------------------------
void FaceIterator::CellFaceIterator::operator++()
{
  ++face_iterator;
}
//-----------------------------------------------------------------------------
bool FaceIterator::CellFaceIterator::end()
{
  return face_iterator.end();
}
//-----------------------------------------------------------------------------
bool FaceIterator::CellFaceIterator::last()
{
  return face_iterator.last();
}
//-----------------------------------------------------------------------------
int FaceIterator::CellFaceIterator::index()
{
  return face_iterator.index();
}
//-----------------------------------------------------------------------------
Face& FaceIterator::CellFaceIterator::operator*() const
{
  return **face_iterator;
}
//-----------------------------------------------------------------------------
Face* FaceIterator::CellFaceIterator::operator->() const
{
  return *face_iterator;
}
//-----------------------------------------------------------------------------
Face* FaceIterator::CellFaceIterator::pointer() const
{
  return *face_iterator;
}
//-----------------------------------------------------------------------------
