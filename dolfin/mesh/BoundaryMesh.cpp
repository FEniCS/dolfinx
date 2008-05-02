// Copyright (C) 2006-2008 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2006-06-21
// Last changed: 2008-05-02

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
BoundaryMesh::BoundaryMesh(Mesh& mesh) : Mesh()
{
  init(mesh);
}
//-----------------------------------------------------------------------------
BoundaryMesh::BoundaryMesh(Mesh& mesh, MeshFunction<uint>& vertex_map)
{
  init(mesh, vertex_map);
}
//-----------------------------------------------------------------------------
BoundaryMesh::BoundaryMesh(Mesh& mesh,
                           MeshFunction<uint>& vertex_map,
                           MeshFunction<uint>& cell_map) : Mesh()
{
  init(mesh, vertex_map, cell_map);
}
//-----------------------------------------------------------------------------
BoundaryMesh::~BoundaryMesh()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
void BoundaryMesh::init(Mesh& mesh)
{
  BoundaryComputation::computeBoundary(mesh, *this);
}
//-----------------------------------------------------------------------------
void BoundaryMesh::init(Mesh& mesh,
                        MeshFunction<uint>& vertex_map)
{
  BoundaryComputation::computeBoundary(mesh, *this, vertex_map);
}
//-----------------------------------------------------------------------------
void BoundaryMesh::init(Mesh& mesh,
                        MeshFunction<uint>& vertex_map,
                        MeshFunction<uint>& cell_map)
{
  BoundaryComputation::computeBoundary(mesh, *this, vertex_map, cell_map);
}
//-----------------------------------------------------------------------------
