// Copyright (C) 2006-2008 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Niclas Jansson 2009.
//
// First added:  2006-06-21
// Last changed: 2009-08-06

#include <iostream>

#include <dolfin/log/log.h>
#include "BoundaryComputation.h"
#include "BoundaryMesh.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
BoundaryMesh::BoundaryMesh() : Mesh()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
BoundaryMesh::BoundaryMesh(const Mesh& mesh) : Mesh()
{
  init(mesh);
}
//-----------------------------------------------------------------------------
BoundaryMesh::~BoundaryMesh()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
void BoundaryMesh::init(const Mesh& mesh)
{
  BoundaryComputation::compute_boundary(mesh, *this);
}
//-----------------------------------------------------------------------------
void BoundaryMesh::init_interior(const Mesh& mesh)
{
  BoundaryComputation::compute_interior_boundary(mesh, *this);
}
//-----------------------------------------------------------------------------
