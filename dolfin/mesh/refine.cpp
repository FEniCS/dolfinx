// Copyright (C) 2010 Garth N. Wells.
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Anders Logg, 2010-2011.
//
// First added:  2010-02-10
// Last changed: 2011-01-11

#include "LocalMeshRefinement.h"
#include "Mesh.h"
#include "MeshFunction.h"
#include "UniformMeshRefinement.h"
#include "refine.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
dolfin::Mesh dolfin::refine(const Mesh& mesh)
{
  Mesh refined_mesh;
  UniformMeshRefinement::refine(refined_mesh, mesh);
  return refined_mesh;
}
//-----------------------------------------------------------------------------
void dolfin::refine(Mesh& refined_mesh, const Mesh& mesh)
{
  UniformMeshRefinement::refine(refined_mesh, mesh);
}
//-----------------------------------------------------------------------------
dolfin::Mesh dolfin::refine(const Mesh& mesh,
                            const MeshFunction<bool>& cell_markers)
{
  Mesh refined_mesh;
  refine(refined_mesh, mesh, cell_markers);
  return refined_mesh;
}
//-----------------------------------------------------------------------------
void dolfin::refine(Mesh& refined_mesh,
                    const Mesh& mesh,
                    const MeshFunction<bool>& cell_markers)
{
  // Count the number of marked cells
  uint n0 = mesh.num_cells();
  uint n = 0;
  for (uint i = 0; i < cell_markers.size(); i++)
    if (cell_markers[i])
      n++;
  info("%d cells out of %d marked for refinement (%.1f%%).",
       n, n0, 100.0 * static_cast<double>(n) / static_cast<double>(n0));

  // Call refinement algorithm
  LocalMeshRefinement::refineRecursivelyByEdgeBisection(refined_mesh,
                                                        mesh,
                                                        cell_markers);

  // Report the number of refined cells
  uint n1 = refined_mesh.num_cells();
  info("Number of cells increased from %d to %d (%.1f%% increase).",
       n0, n1, 100.0 * (static_cast<double>(n1) / static_cast<double>(n0) - 1.0));
}
//-----------------------------------------------------------------------------
