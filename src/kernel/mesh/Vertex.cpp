// Copyright (C) 2002-2006 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// First added:  2002
// Last changed: 2006-03-10

#include <dolfin/Vertex.h>
#include <dolfin/GenericCell.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
Vertex::Vertex()
{
  clear();
}
//-----------------------------------------------------------------------------
Vertex::Vertex(real x)
{
  clear();
  p.x = x;
}
//-----------------------------------------------------------------------------
Vertex::Vertex(real x, real y)
{
  clear();
  p.x = x;
  p.y = y;
}
//-----------------------------------------------------------------------------
Vertex::Vertex(real x, real y, real z)
{
  clear();
  p.x = x;
  p.y = y;
  p.z = z;
}
//-----------------------------------------------------------------------------
Vertex::~Vertex()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
void Vertex::clear()
{
  _mesh = 0;
  _id = -1;

  p.x = 0.0;
  p.y = 0.0;
  p.z = 0.0;

  nn.clear();
  nc.clear();
  ne.clear();

  _parent = 0;
  _child = 0;
}
//-----------------------------------------------------------------------------
int Vertex::id() const
{
  return _id;
}
//-----------------------------------------------------------------------------
int Vertex::numVertexNeighbors() const
{
  return nn.size();
}
//-----------------------------------------------------------------------------
int Vertex::numCellNeighbors() const
{
  return nc.size();
}
//-----------------------------------------------------------------------------
int Vertex::numEdgeNeighbors() const
{
  return ne.size();
}
//-----------------------------------------------------------------------------
Vertex& Vertex::vertex(int i) const
{
  return *nn(i);
}
//-----------------------------------------------------------------------------
Cell& Vertex::cell(int i) const
{
  return *nc(i);
}
//-----------------------------------------------------------------------------
Edge& Vertex::edge(int i) const
{
  return *ne(i);
}
//-----------------------------------------------------------------------------
Vertex* Vertex::parent() const 
{
  return _parent;
}
//-----------------------------------------------------------------------------
Vertex* Vertex::child() const
{
  return _child;
}
//-----------------------------------------------------------------------------
Mesh& Vertex::mesh()
{
  return *_mesh;
}
//-----------------------------------------------------------------------------
const Mesh& Vertex::mesh() const
{
  return *_mesh;
}
//-----------------------------------------------------------------------------
Point& Vertex::coord()
{
  return p;
}
//-----------------------------------------------------------------------------
Point Vertex::coord() const
{
  return p;
}
//-----------------------------------------------------------------------------
Point Vertex::midpoint(const Vertex& n) const
{
  return p.midpoint(n.p);
}
//-----------------------------------------------------------------------------
real Vertex::dist(const Vertex& n) const
{
  return p.dist(n.p);
}
//-----------------------------------------------------------------------------
real Vertex::dist(const Point& p) const
{
  return this->p.dist(p);
}
//-----------------------------------------------------------------------------
real Vertex::dist(real x, real y, real z) const
{
  return p.dist(x, y, z);
}
//-----------------------------------------------------------------------------
bool Vertex::neighbor(const Vertex& n) const
{
  for (VertexIterator neighbor(*this); !neighbor.end(); ++neighbor)
    if ( &n == neighbor )
      return true;
  
  return false;
}
//-----------------------------------------------------------------------------
bool Vertex::operator==(const Vertex& vertex) const
{
  return this == &vertex;
}
//-----------------------------------------------------------------------------
bool Vertex::operator!=(const Vertex& vertex) const
{
  return this != &vertex;
}
//-----------------------------------------------------------------------------
bool Vertex::operator== (int id) const
{
  return _id == id;
}
//-----------------------------------------------------------------------------
bool Vertex::operator< (int id) const
{
  return _id < id;
}
//-----------------------------------------------------------------------------
bool Vertex::operator<= (int id) const
{
  return _id <= id;
}
//-----------------------------------------------------------------------------
bool Vertex::operator> (int id) const
{
  return _id > id;
}
//-----------------------------------------------------------------------------
bool Vertex::operator>= (int id) const
{
  return _id >= id;
}
//-----------------------------------------------------------------------------
bool dolfin::operator== (int id, const Vertex& vertex)
{
  return vertex == id;
}
//-----------------------------------------------------------------------------
bool dolfin::operator< (int id, const Vertex& vertex)
{
  return vertex > id;
}
//-----------------------------------------------------------------------------
bool dolfin::operator<= (int id, const Vertex& vertex)
{
  return vertex >= id;
}
//-----------------------------------------------------------------------------
bool dolfin::operator> (int id, const Vertex& vertex)
{
  return vertex < id;
}
//-----------------------------------------------------------------------------
bool dolfin::operator>= (int id, const Vertex& vertex)
{
  return vertex <= id;
}
//-----------------------------------------------------------------------------
dolfin::LogStream& dolfin::operator<<(LogStream& stream, const Vertex& vertex)
{
  int id = vertex.id();
  const Point p = vertex.coord();
  
  stream << "[ Vertex: id = " << id
	 << " x = (" << p.x << "," << p.y << "," << p.z << ") ]";
  
  return stream;
}
//-----------------------------------------------------------------------------
int Vertex::setID(int id, Mesh& mesh)
{
  _mesh = &mesh;
  return _id = id;
}
//-----------------------------------------------------------------------------
void Vertex::setMesh(Mesh& mesh)
{
  _mesh = &mesh;
}
//-----------------------------------------------------------------------------
void Vertex::setParent(Vertex& parent)
{
  // Set parent vertex: a vertex is parent to if it has the same coordinates 
  // and is contained in the next coarser mesh 
  this->_parent = &parent;
}
//-----------------------------------------------------------------------------
void Vertex::setChild(Vertex& child)
{
  // Set child vertex: a vertex is child to if it has the same coordinates 
  // and is contained in the next finer mesh 
  this->_child = &child;
}
//-----------------------------------------------------------------------------
void Vertex::removeParent(Vertex& parent)
{
  // Remove parent
  this->_parent = 0;
}
//-----------------------------------------------------------------------------
void Vertex::removeChild()
{
  // Remove child 
  this->_child = 0;
}
//-----------------------------------------------------------------------------
void Vertex::set(real x, real y, real z)
{
  p.x = x;
  p.y = y;
  p.z = z;
}
//-----------------------------------------------------------------------------
