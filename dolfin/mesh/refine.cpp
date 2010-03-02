// Copyright (C) 2010 Garth N. Wells.
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Anders Logg, 2010.
//
// First added:  2010-02-10
// Last changed: 2010-02-26

#include "LocalMeshRefinement.h"
#include "Mesh.h"
#include "MeshFunction.h"
#include "UniformMeshRefinement.h"
#include "refine.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
dolfin::Mesh dolfin::refine(const Mesh& mesh)
{
  info("Calling uniform refinement free function: mesh = %x", &mesh);
  Mesh refined_mesh;
  UniformMeshRefinement::refine(refined_mesh, mesh);
  return refined_mesh;
}
//-----------------------------------------------------------------------------
void dolfin::refine(Mesh& refined_mesh, const Mesh& mesh)
{
  info("Calling uniform refinement free function with output argument: mesh = %x", &mesh);
  UniformMeshRefinement::refine(refined_mesh, mesh);
}
//-----------------------------------------------------------------------------
dolfin::Mesh dolfin::refine(const Mesh& mesh,
                            const MeshFunction<bool>& cell_markers)
{
  info("Calling uniform refinement free function: mesh = %x", &mesh);
  Mesh refined_mesh;
  LocalMeshRefinement::refineRecursivelyByEdgeBisection(refined_mesh,
                                                        mesh,
                                                        cell_markers);
  return refined_mesh;
}
//-----------------------------------------------------------------------------
void dolfin::refine(Mesh& refined_mesh,
                    const Mesh& mesh,
                    const MeshFunction<bool>& cell_markers)
{
  info("Calling uniform refinement free function with output argument: mesh = %x", &mesh);
  LocalMeshRefinement::refineRecursivelyByEdgeBisection(refined_mesh,
                                                        mesh,
                                                        cell_markers);
}
//-----------------------------------------------------------------------------
