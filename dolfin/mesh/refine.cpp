// Copyright (C) 2010 Garth N. Wells.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2010-02-10
// Last changed:

#include "LocalMeshRefinement.h"
#include "Mesh.h"
#include "MeshFunction.h"
#include "UniformMeshRefinement.h"
#include "refine.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
dolfin::Mesh refine(const Mesh& mesh)
{
  return UniformMeshRefinement::refine(mesh);
}
//-----------------------------------------------------------------------------
dolfin::Mesh refine(const Mesh& mesh, const MeshFunction<bool>& cell_markers)
{
  return LocalMeshRefinement::refineRecursivelyByEdgeBisection(mesh, cell_markers);
}
//-----------------------------------------------------------------------------
