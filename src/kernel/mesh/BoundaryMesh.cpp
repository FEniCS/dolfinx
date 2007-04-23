// Copyright (C) 2006 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2006-06-21
// Last changed: 2006-10-16

#include <dolfin/BoundaryComputation.h>
#include <dolfin/BoundaryMesh.h>

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
                        MeshFunction<uint>& vertex_map,
                        MeshFunction<uint>& cell_map)
{
  BoundaryComputation::computeBoundary(mesh, *this, vertex_map, cell_map);
}
//-----------------------------------------------------------------------------
