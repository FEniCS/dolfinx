#include <dolfin/Node.hh>
#include <dolfin/Triangle.hh>
#include <dolfin/Tetrahedron.hh>
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
Triangle* GridData::createTriangle()
{
  int id;
  Triangle *t = triangles.create(&id);
  t->setID(id);
  return t;
}
//-----------------------------------------------------------------------------
Tetrahedron* GridData::createTetrahedron()
{
  int id;
  Tetrahedron *t = tetrahedrons.create(&id);
  t->setID(id);
  return t;
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
Triangle* GridData::createTriangle(int n0, int n1, int n2)
{
  int id;
  Triangle *t = triangles.create(&id);
  t->setID(id);
  t->set(getNode(n0),getNode(n1),getNode(n2));
  return t;
}
//-----------------------------------------------------------------------------
Tetrahedron* GridData::createTetrahedron(int n0, int n1, int n2, int n3)
{
  int id;
  Tetrahedron *t = tetrahedrons.create(&id);
  t->setID(id);
  t->set(getNode(n0),getNode(n1),getNode(n2),getNode(n3));
  return t;
}
//-----------------------------------------------------------------------------
Node* GridData::getNode(int id)
{
  return nodes.pointer(id);
}
//-----------------------------------------------------------------------------
Triangle* GridData::getTriangle(int id)
{
  return triangles.pointer(id);
}
//-----------------------------------------------------------------------------
Tetrahedron* GridData::getTetrahedron(int id)
{
  return tetrahedrons.pointer(id);
}
//-----------------------------------------------------------------------------
