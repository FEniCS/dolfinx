// Copyright (C) 2003 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#include <dolfin/Grid.h>
#include <dolfin/Boundary.h>
#include <dolfin/Cell.h>
#include <dolfin/GenericCell.h>
#include <dolfin/FaceIterator.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
// FaceIterator
//-----------------------------------------------------------------------------
FaceIterator::FaceIterator(const Grid &grid)
{
  e = new GridFaceIterator(grid);
}
//-----------------------------------------------------------------------------
FaceIterator::FaceIterator(const Grid *grid)
{
  e = new GridFaceIterator(*grid);
}
//-----------------------------------------------------------------------------
FaceIterator::FaceIterator(const Boundary& boundary)
{
  e = new BoundaryFaceIterator(boundary);
}
//-----------------------------------------------------------------------------
FaceIterator::FaceIterator(const Cell& cell)
{
  e = new CellFaceIterator(cell);
}
//-----------------------------------------------------------------------------
FaceIterator::FaceIterator(const CellIterator& cellIterator)
{
  e = new CellFaceIterator(*cellIterator);
}
//-----------------------------------------------------------------------------
FaceIterator::operator FacePointer() const
{
  return e->pointer();
}
//-----------------------------------------------------------------------------
FaceIterator::~FaceIterator()
{
  delete e;
}
//-----------------------------------------------------------------------------
FaceIterator& FaceIterator::operator++()
{
  ++(*e);

  return *this;
}
//-----------------------------------------------------------------------------
bool FaceIterator::end()
{
  return e->end();
}
//-----------------------------------------------------------------------------
bool FaceIterator::last()
{
  return e->last();
}
//-----------------------------------------------------------------------------
int FaceIterator::index()
{
  return e->index();
}
//-----------------------------------------------------------------------------
Face& FaceIterator::operator*() const
{
  return *(*e);
}
//-----------------------------------------------------------------------------
Face* FaceIterator::operator->() const
{
  return e->pointer();
}
//-----------------------------------------------------------------------------
bool FaceIterator::operator==(const FaceIterator& e) const
{
  return this->e->pointer() == e.e->pointer();
}
//-----------------------------------------------------------------------------
bool FaceIterator::operator!=(const FaceIterator& e) const
{
  return this->e->pointer() != e.e->pointer();
}
//-----------------------------------------------------------------------------
// FaceIterator::GridFaceIterator
//-----------------------------------------------------------------------------
FaceIterator::GridFaceIterator::GridFaceIterator(const Grid& grid)
{
  face_iterator = grid.gd->faces.begin();
  at_end = grid.gd->faces.end();
}
//-----------------------------------------------------------------------------
void FaceIterator::GridFaceIterator::operator++()
{
  ++face_iterator;
}
//-----------------------------------------------------------------------------
bool FaceIterator::GridFaceIterator::end()
{
  return face_iterator == at_end;
}
//-----------------------------------------------------------------------------
bool FaceIterator::GridFaceIterator::last()
{
  return face_iterator.last();
}
//-----------------------------------------------------------------------------
int FaceIterator::GridFaceIterator::index()
{
  return face_iterator.index();
}
//-----------------------------------------------------------------------------
Face& FaceIterator::GridFaceIterator::operator*() const
{
  return *face_iterator;
}
//-----------------------------------------------------------------------------
Face* FaceIterator::GridFaceIterator::operator->() const
{
  return face_iterator.pointer();
}
//-----------------------------------------------------------------------------
Face* FaceIterator::GridFaceIterator::pointer() const
{
  return face_iterator.pointer();
}
//-----------------------------------------------------------------------------
// FaceIterator::BoundaryFaceIterator
//-----------------------------------------------------------------------------
FaceIterator::BoundaryFaceIterator::BoundaryFaceIterator
(const Boundary& boundary) : face_iterator(boundary.grid->bd->faces)
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
