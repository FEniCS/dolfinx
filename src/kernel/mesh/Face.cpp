// Copyright (C) 2003-2006 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// Modified by Garth N. Wells, 2006.
//
// First added:  2003
// Last changed: 2006-05-03

#include <dolfin/Point.h>
#include <dolfin/Vertex.h>
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
  _mesh = 0;
  _id = -1;
  fe.clear();
}
//-----------------------------------------------------------------------------
int Face::id() const
{
  return _id;
}
//-----------------------------------------------------------------------------
unsigned int Face::numEdges() const
{
  return fe.size();
}
//-----------------------------------------------------------------------------
unsigned int Face::numCellNeighbors() const
{
  return fc.size();
}
//-----------------------------------------------------------------------------
Edge& Face::edge(int i) const
{
  return *fe(i);
}
//-----------------------------------------------------------------------------
Cell& Face::cell(int i) const
{
  return *fc(i);
}
//-----------------------------------------------------------------------------
int Face::localID(int i) const
{
  return f_local_id(i);
}
//-----------------------------------------------------------------------------
Mesh& Face::mesh()
{
  return *_mesh;
}
//-----------------------------------------------------------------------------
const Mesh& Face::mesh() const
{
  return *_mesh;
}
//-----------------------------------------------------------------------------
bool Face::equals(const Edge& e0, const Edge& e1, const Edge& e2) const
{
  dolfin_assert(fe.size() == 3);

  // Only two edges need to be checked for identification
  return equals(e0,e1);
}
//-----------------------------------------------------------------------------
bool Face::equals(const Edge& e0, const Edge& e1) const
{
  dolfin_assert(fe.size() == 3);

  if ( fe(0) == &e0 && fe(1) == &e1 )
    return true;

  if ( fe(0) == &e0 && fe(2) == &e1 )
    return true;

  if ( fe(1) == &e0 && fe(0) == &e1 )
    return true;

  if ( fe(1) == &e0 && fe(2) == &e1 )
    return true;

  if ( fe(2) == &e0 && fe(0) == &e1 )
    return true;

  if ( fe(2) == &e0 && fe(1) == &e1 )
    return true;

  return false;
}
//-----------------------------------------------------------------------------
bool Face::contains(const Vertex& n) const
{
  return fe(0)->contains(n) || fe(1)->contains(n) || fe(2)->contains(n);
}
//-----------------------------------------------------------------------------
bool Face::contains(const Point& point) const
{
  // Quick check: return true if the point is in the same plane by
  // checking the size of the scalar product with the normal of the plain
  // obtained from the cross product (should be zero)
  
  const Edge* e0 = fe(0);
  const Edge* e1 = fe(1);

  Point v0 = e0->n1->coord() - e0->n0->coord();
  Point v1 = e1->n1->coord() - e1->n0->coord();

  Point n = v0.cross(v1);
  Point v = point - e0->n0->coord();

  return std::abs(n*v) < DOLFIN_EPS;
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
int Face::setID(int id, Mesh& mesh)
{
  _mesh = &mesh;
  return _id = id;
}
//-----------------------------------------------------------------------------
void Face::setMesh(Mesh& mesh)
{
  _mesh = &mesh;
}
//-----------------------------------------------------------------------------
void Face::set(Edge& e0, Edge& e1, Edge& e2)
{
  fe.init(3);
  fe(0) = &e0;
  fe(1) = &e1;
  fe(2) = &e2;
}
//-----------------------------------------------------------------------------
