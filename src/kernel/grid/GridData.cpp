#include <dolfin/Node.h>
#include <dolfin/Triangle.h>
#include <dolfin/Tetrahedron.h>
#include <dolfin/Cell.h>
#include "GridData.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
GridData::GridData()
{


  
}
//-----------------------------------------------------------------------------
GridData::~GridData()
{

}
//-----------------------------------------------------------------------------
Node* GridData::createNode()
{
  int id;
  Node *n = nodes.create(&id);
  n->setID(id);
  return n;
}
//-----------------------------------------------------------------------------
Cell* GridData::createCell(Cell::Type type)
{
  int id;
  Cell *c = cells.create(&id);
  c->setID(id);
  c->init(type);
  return c;
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
Cell* GridData::createCell(Cell::Type type, int n0, int n1, int n2)
{
  int id;
  Cell *c = cells.create(&id);
  c->init(type);
  c->setID(id);
  c->set(getNode(n0),getNode(n1),getNode(n2));
  return c;
}
//-----------------------------------------------------------------------------
Cell* GridData::createCell(Cell::Type type, int n0, int n1, int n2, int n3)
{
  int id;
  Cell *c = cells.create(&id);
  c->init(type);
  c->setID(id);
  c->set(getNode(n0),getNode(n1),getNode(n2),getNode(n3));
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
