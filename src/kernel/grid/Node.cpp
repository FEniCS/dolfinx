#include <dolfin/Node.h>
#include <dolfin/GenericCell.h>
#include <dolfin/ShortList.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
Node::Node()
{
  _id = -1;
  _boundary = -1;
}
//-----------------------------------------------------------------------------
Node::Node(real x)
{
  _id = -1;
  _boundary = -1;

  p.x = x;
}
//-----------------------------------------------------------------------------
Node::Node(real x, real y)
{
  _id = -1;
  _boundary = -1;

  p.x = x;
  p.y = y;
}
//-----------------------------------------------------------------------------
Node::Node(real x, real y, real z)
{
  _id = -1;
  _boundary = -1;

  p.x = x;
  p.y = y;
  p.z = z;
}
//-----------------------------------------------------------------------------
void Node::set(real x, real y, real z)
{
  p.x = x;
  p.y = y;
  p.z = z;
}
//-----------------------------------------------------------------------------
int Node::id() const
{
  return _id;
}
//-----------------------------------------------------------------------------
Point Node::coord() const
{
  return p;
}
//-----------------------------------------------------------------------------
int Node::boundary() const
{
  return _boundary;
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
int Node::setID(int id)
{
  return _id = id;
}
//-----------------------------------------------------------------------------
// Additional operators
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
