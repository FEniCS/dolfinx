// Copyright (C) 2003-2005 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// First added:  2003
// Last changed: 2005-12-01

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
int Boundary::noVertices() const
{
  dolfin_assert(mesh);
  return mesh->bd->noVertices();
}
//-----------------------------------------------------------------------------
int Boundary::noEdges() const
{
  dolfin_assert(mesh);
  return mesh->bd->noEdges();
}
//-----------------------------------------------------------------------------
int Boundary::noFaces() const
{
  dolfin_assert(mesh);
  return mesh->bd->noFaces();
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
