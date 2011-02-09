// Copyright (C) 2011 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2011-02-07
// Last changed: 2011-02-07

#include <dolfin/parameter/GlobalParameters.h>
#include <dolfin/log/log.h>
#include <dolfin/mesh/Mesh.h>
#include <dolfin/mesh/MeshFunction.h>
#include "BisectionRefinement.h"
#include "RegularCutRefinement.h"
#include "LocalMeshRefinement.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
void LocalMeshRefinement::refine(Mesh& refined_mesh,
                                 const Mesh& mesh,
                                 const MeshFunction<bool>& cell_markers)
{
  // Count the number of marked cells
  const uint n0 = mesh.num_cells();
  uint n = 0;
  for (uint i = 0; i < cell_markers.size(); i++)
    if (cell_markers[i])
      n++;
  info("%d cells out of %d marked for refinement (%.1f%%).",
       n, n0, 100.0 * static_cast<double>(n) / static_cast<double>(n0));

  // Call refinement algorithm
  const std::string refinement_algorithm = parameters["refinement_algorithm"];
  if (refinement_algorithm == "bisection")
    BisectionRefinement::refine_by_bisection(refined_mesh, mesh, cell_markers);
  else if (refinement_algorithm == "iterative_bisection")
    BisectionRefinement::refine_by_iterative_bisection(refined_mesh, mesh, cell_markers);
  else if (refinement_algorithm == "recursive_bisection")
    BisectionRefinement::refine_by_recursive_bisection(refined_mesh, mesh, cell_markers);
  else if (refinement_algorithm == "regular_cut")
    RegularCutRefinement::refine(refined_mesh, mesh, cell_markers);
  else
  {
    error("Unknown local mesh refinement algorithm: %s", refinement_algorithm.c_str());
  }

  // Report the number of refined cells
  if (refined_mesh.topology().dim() > 0)
  {
    const uint n1 = refined_mesh.num_cells();
    info("Number of cells increased from %d to %d (%.1f%% increase).",
         n0, n1, 100.0 * (static_cast<double>(n1) / static_cast<double>(n0) - 1.0));
  }
  else
    info("Refined mesh is empty.");
}
//-----------------------------------------------------------------------------
