// Copyright (C) 2003 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#include <dolfin/Edge.h>
#include <dolfin/Face.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
Face::Face()
{
  clear();
}
//-----------------------------------------------------------------------------
Face::~Face()
{
  clear();
}
//-----------------------------------------------------------------------------
void Face::clear()
{
  grid = 0;
  _id = -1;
  fe.clear();
}
//-----------------------------------------------------------------------------
int Face::id() const
{
  return _id;
}
//-----------------------------------------------------------------------------
int Face::size() const
{
  return fe.size();
}
//-----------------------------------------------------------------------------
Edge* Face::edge(int i) const
{
  return fe(i);
}
//-----------------------------------------------------------------------------
bool Face::equals(Edge* e0, Edge* e1, Edge* e2) const
{
  dolfin_assert(fe.size() == 3);

  if ( fe(0) == e0 && fe(1) == e1 && fe(2) == e2 )
    return true;

  if ( fe(0) == e0 && fe(1) == e2 && fe(2) == e1 )
    return true;

  if ( fe(0) == e1 && fe(1) == e0 && fe(2) == e2 )
    return true;

  if ( fe(0) == e1 && fe(1) == e2 && fe(2) == e0 )
    return true;

  if ( fe(0) == e2 && fe(1) == e0 && fe(2) == e1 )
    return true;

  if ( fe(0) == e2 && fe(1) == e1 && fe(2) == e0 )
    return true;

  return false;
}
//-----------------------------------------------------------------------------
dolfin::LogStream& dolfin::operator<<(LogStream& stream, const Face& face)
{
  stream << "[ Face: id = " << face.id() << " edges = ( ";
  for (EdgeIterator e(face); !e.end(); ++e)
    stream << e->id() << " ";
  stream << ") ]";
  
  return stream;
}
//-----------------------------------------------------------------------------
int Face::setID(int id, Grid* grid)
{
  this->grid = grid;
  return _id = id;
}
//-----------------------------------------------------------------------------
void Face::set(Edge* e0, Edge* e1, Edge* e2)
{
  fe.init(3);
  fe(0) = e0;
  fe(1) = e1;
  fe(2) = e2;
}
//-----------------------------------------------------------------------------
