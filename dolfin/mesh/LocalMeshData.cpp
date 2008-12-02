// Copyright (C) 2008 Ola Skavhaug.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2008-11-28
// Last changed: 2008-12-02
//
// Modified by Anders Logg, 2008.

#include "LocalMeshData.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
LocalMeshData::LocalMeshData()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
LocalMeshData::~LocalMeshData()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
void LocalMeshData::clear()
{
  vertex_coordinates.clear();
  vertex_indices.clear();
  cell_vertices.clear();
}
//-----------------------------------------------------------------------------
