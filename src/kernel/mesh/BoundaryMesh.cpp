// Copyright (C) 2006 Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// First added:  2006-06-21
// Last changed: 2006-06-22

#include <dolfin/BoundaryComputation.h>
#include <dolfin/BoundaryMesh.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
BoundaryMesh::BoundaryMesh() : NewMesh()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
BoundaryMesh::BoundaryMesh(NewMesh& mesh) : NewMesh()
{
  init(mesh);
}
//-----------------------------------------------------------------------------
BoundaryMesh::~BoundaryMesh()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
void BoundaryMesh::init(NewMesh& mesh)
{
  BoundaryComputation::computeBoundary(mesh, *this);
}
//-----------------------------------------------------------------------------
