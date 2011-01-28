// Copyright (C) 2010 Garth N. Wells.
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Anders Logg, 2010-2011.
//
// First added:  2010-02-10
// Last changed: 2011-01-28

#include <boost/shared_ptr.hpp>

#include <dolfin/common/NoDeleter.h>
#include <dolfin/mesh/LocalMeshRefinement.h>
#include <dolfin/mesh/Mesh.h>
#include <dolfin/mesh/MeshEntity.h>
#include <dolfin/mesh/MeshFunction.h>
#include <dolfin/mesh/UniformMeshRefinement.h>
#include <dolfin/function/FunctionSpace.h>
#include <dolfin/fem/FiniteElement.h>
#include <dolfin/fem/DofMap.h>
#include "refine.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
dolfin::Mesh dolfin::refine(const Mesh& mesh)
{
  Mesh refined_mesh;
  refine(refined_mesh, mesh);
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
dolfin::FunctionSpace dolfin::refine(const FunctionSpace& V)
{
  // Refine mesh
  const Mesh& mesh = V.mesh();
  boost::shared_ptr<Mesh> refined_mesh(new Mesh());
  refine(*refined_mesh, mesh);

  // Refine space
  FunctionSpace W = refine(V, *refined_mesh);

  return W;
}
//-----------------------------------------------------------------------------
dolfin::FunctionSpace dolfin::refine(const FunctionSpace& V,
                                     const MeshFunction<bool>& cell_markers)
{
  // Refine mesh
  const Mesh& mesh = V.mesh();
  boost::shared_ptr<Mesh> refined_mesh(new Mesh());
  refine(*refined_mesh, mesh, cell_markers);

  // Refine space
  FunctionSpace W = refine(V, *refined_mesh);

  return W;
}
//-----------------------------------------------------------------------------
dolfin::FunctionSpace dolfin::refine(const FunctionSpace& V,
                                     const Mesh& refined_mesh)
{
#ifndef UFC_DEV
  info("UFC_DEV compiler flag is not set.");
  error("Refinement of function spaces relies on the development version of UFC.");
  return V;
#else

  // Get DofMap (GenericDofMap does not know about ufc::dof_map)
  const DofMap* dofmap = dynamic_cast<const DofMap*>(&V.dofmap());
  if (!dofmap)
  {
    info("FunctionSpace is defined by a non-stand dofmap.");
    error("Unable to refine function space.");
  }

  // Create new copies of UFC finite element and dofmap
  boost::shared_ptr<ufc::finite_element> ufc_element(V.element().ufc_element()->create());
  boost::shared_ptr<ufc::dof_map> ufc_dofmap(dofmap->ufc_dofmap()->create());

  // Create DOLFIN finite element and dofmap
  boost::shared_ptr<FiniteElement> refined_element(new FiniteElement(ufc_element));
  boost::shared_ptr<DofMap> refined_dofmap(new DofMap(ufc_dofmap, refined_mesh));

  // Create new function space
  FunctionSpace W(reference_to_no_delete_pointer(refined_mesh),
                  refined_element,
                  refined_dofmap);

  return W;

#endif
}
//-----------------------------------------------------------------------------
