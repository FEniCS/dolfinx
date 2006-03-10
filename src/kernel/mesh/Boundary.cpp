// Copyright (C) 2003-2006 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// First added:  2003
// Last changed: 2006-03-10

#include <dolfin/dolfin_log.h>
#include <dolfin/Mesh.h>
#include <dolfin/BoundaryInit.h>
#include <dolfin/Boundary.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
Boundary::Boundary()
{
  this->mesh = 0;
}
//-----------------------------------------------------------------------------
Boundary::Boundary(Mesh& mesh)
{
  this->mesh = &mesh;
  init();
}
//-----------------------------------------------------------------------------
Boundary::~Boundary()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
int Boundary::numVertices() const
{
  dolfin_assert(mesh);
  return mesh->bd->numVertices();
}
//-----------------------------------------------------------------------------
int Boundary::numEdges() const
{
  dolfin_assert(mesh);
  return mesh->bd->numEdges();
}
//-----------------------------------------------------------------------------
int Boundary::numFaces() const
{
  dolfin_assert(mesh);
  return mesh->bd->numFaces();
}
//-----------------------------------------------------------------------------
void Boundary::init()
{
  if ( mesh->bd->empty() )
    BoundaryInit::init(*mesh);
}
//-----------------------------------------------------------------------------
void Boundary::clear()
{
  mesh->bd->clear();
}
//-----------------------------------------------------------------------------
