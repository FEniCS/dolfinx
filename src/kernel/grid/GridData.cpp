#include <dolfin/Node.h>
#include <dolfin/Triangle.h>
#include <dolfin/Tetrahedron.h>
#include <dolfin/Cell.h>
#include <dolfin/GridData.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
Node* GridData::createNode()
{ 
  int id;
  Node *n = nodes.create(&id);
  n->setID(id);
  return n;
}
//-----------------------------------------------------------------------------
Node* GridData::createNode(real x, real y, real z)
{
  int id;
  Node *n = nodes.create(&id);
  n->setID(id);
  n->set(x,y,z);  
  return n;
}
//-----------------------------------------------------------------------------
Cell* GridData::createCell(int level, Cell::Type type)
{
  int id;
  Cell *c = cells.create(&id);
  c->setLevel(level);
  c->setID(id);
  c->init(type);
  return c;
}
//-----------------------------------------------------------------------------
Cell* GridData::createCell(int level, Cell::Type type, int n0, int n1, int n2)
{
  int id;
  Cell *c = cells.create(&id);
  c->setLevel(level);
  c->init(type);
  c->setID(id);
  c->set(getNode(n0),getNode(n1),getNode(n2));
  return c;
}
//-----------------------------------------------------------------------------
Cell* GridData::createCell(int level, Cell::Type type, int n0, int n1, int n2, int n3)
{
  int id;
  Cell *c = cells.create(&id);
  c->setLevel(level);
  c->init(type);
  c->setID(id);
  c->set(getNode(n0),getNode(n1),getNode(n2),getNode(n3));
  return c;
}
//-----------------------------------------------------------------------------
Cell* GridData::createCell(int level, Cell::Type type, Node* n0, Node* n1, Node* n2)
{
  int id;
  Cell *c = cells.create(&id);
  c->setLevel(level);
  c->init(type);
  c->setID(id);
  c->set(n0,n1,n2);
  return c;
}
//-----------------------------------------------------------------------------
Cell* GridData::createCell(int level, Cell::Type type, Node* n0, Node* n1, Node* n2, Node* n3)
{
  int id;
  Cell *c = cells.create(&id);
  c->setLevel(level);
  c->init(type);
  c->setID(id);
  c->set(n0,n1,n2,n3);
  return c;
}
//-----------------------------------------------------------------------------
Node* GridData::getNode(int id)
{
  return nodes.pointer(id);
}
//-----------------------------------------------------------------------------
Cell* GridData::getCell(int id)
{
  return cells.pointer(id);
}
//-----------------------------------------------------------------------------
int GridData::noNodes() const
{
  return nodes.size();
}
//-----------------------------------------------------------------------------
int GridData::noCells() const
{
  return cells.size();
}
//-----------------------------------------------------------------------------
