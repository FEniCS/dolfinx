// Copyright (C) 2002 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#include <dolfin/Node.h>
#include <dolfin/GenericCell.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
Node::Node()
{
  grid = 0;
  _id = -1;

  // FIXME: Remove?
  _boundary = -1;
  _child = NULL;
}
//-----------------------------------------------------------------------------
Node::Node(real x)
{
  grid = 0;
  _id = -1;

  p.x = x;

  // FIXME: Remove?
  _boundary = -1;
  _child = NULL;
}
//-----------------------------------------------------------------------------
Node::Node(real x, real y)
{
  grid = 0;
  _id = -1;

  p.x = x;
  p.y = y;
  
  // FIXME: Remove?
  _boundary = -1;
  _child = NULL;
}
//-----------------------------------------------------------------------------
Node::Node(real x, real y, real z)
{
  grid = 0;
  _id = -1;

  p.x = x;
  p.y = y;
  p.z = z;

  // FIXME: Remove?
  _boundary = -1;
  _child = NULL;
}
//-----------------------------------------------------------------------------
int Node::id() const
{
  return _id;
}
//-----------------------------------------------------------------------------
int Node::noNodeNeighbors() const
{
  return nn.size();
}
//-----------------------------------------------------------------------------
int Node::noCellNeighbors() const
{
  return nc.size();
}
//-----------------------------------------------------------------------------
int Node::noEdgeNeighbors() const
{
  return ne.size();
}
//-----------------------------------------------------------------------------
Node* Node::node(int i) const
{
  return nn(i);
}
//-----------------------------------------------------------------------------
Cell* Node::cell(int i) const
{
  return nc(i);
}
//-----------------------------------------------------------------------------
Edge* Node::edge(int i) const
{
  return ne(i);
}
//-----------------------------------------------------------------------------
Point Node::coord() const
{
  return p;
}
//-----------------------------------------------------------------------------
real Node::dist(const Node& n) const
{
  return p.dist(n.p);
}
//-----------------------------------------------------------------------------
bool Node::neighbor(Node* n)
{
  for (NodeIterator neighbor(*this); !neighbor.end(); ++neighbor)
    if ( n == neighbor )
      return true;
  
  return false;
}
//-----------------------------------------------------------------------------
int Node::boundary() const
{
  return _boundary;
}
//-----------------------------------------------------------------------------
bool Node::operator== (int id) const
{
  return _id == id;
}
//-----------------------------------------------------------------------------
bool Node::operator< (int id) const
{
  return _id < id;
}
//-----------------------------------------------------------------------------
bool Node::operator<= (int id) const
{
  return _id <= id;
}
//-----------------------------------------------------------------------------
bool Node::operator> (int id) const
{
  return _id > id;
}
//-----------------------------------------------------------------------------
bool Node::operator>= (int id) const
{
  return _id >= id;
}
//-----------------------------------------------------------------------------
bool dolfin::operator== (int id, const Node& node)
{
  return node == id;
}
//-----------------------------------------------------------------------------
bool dolfin::operator< (int id, const Node& node)
{
  return node > id;
}
//-----------------------------------------------------------------------------
bool dolfin::operator<= (int id, const Node& node)
{
  return node >= id;
}
//-----------------------------------------------------------------------------
bool dolfin::operator> (int id, const Node& node)
{
  return node < id;
}
//-----------------------------------------------------------------------------
bool dolfin::operator>= (int id, const Node& node)
{
  return node <= id;
}
//-----------------------------------------------------------------------------
dolfin::LogStream& dolfin::operator<<(LogStream& stream, const Node& node)
{
  int id = node.id();
  Point p = node.coord();
  
  stream << "[ Node: id = " << id
	 << " x = (" << p.x << "," << p.y << "," << p.z << ") ]";
  
  return stream;
}
//-----------------------------------------------------------------------------
int Node::setID(int id, Grid* grid)
{
  this->grid = grid;
  return _id = id;
}
//-----------------------------------------------------------------------------
void Node::set(real x, real y, real z)
{
  p.x = x;
  p.y = y;
  p.z = z;
}
//-----------------------------------------------------------------------------





// FIXME: Remove?


//-----------------------------------------------------------------------------
void Node::setMarkedForReUse(bool re_use)
{
  _marked_for_re_use = re_use;
}
//-----------------------------------------------------------------------------
bool Node::markedForReUse()
{
  return _marked_for_re_use;
}
//-----------------------------------------------------------------------------
int Node::level() const
{
  return _level;
}
//-----------------------------------------------------------------------------
void Node::setLevel(int level)
{
  _level = level;
}
//-----------------------------------------------------------------------------

Node* Node::child() 
{
  return _child;
}
//-----------------------------------------------------------------------------
void Node::setChild(Node* child) 
{
  _child = child;
}
//-----------------------------------------------------------------------------
