// Copyright (C) 2007-2010 Garth N. Wells.
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Ola Skavhaug, 2007.
// Modified by Anders Logg, 2008-2009.
//
// First added:  2007-05-24
// Last changed: 2010-12-29

#include <boost/scoped_array.hpp>
#include <dolfin/mesh/Cell.h>
#include <dolfin/mesh/Facet.h>
#include <dolfin/mesh/Mesh.h>
#include <dolfin/la/GenericSparsityPattern.h>
#include <dolfin/function/FunctionSpace.h>
#include "SparsityPatternBuilder.h"
#include "GenericDofMap.h"
#include "UFC.h"
#include "Form.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
void SparsityPatternBuilder::build(GenericSparsityPattern& sparsity_pattern,
                                   const Mesh& mesh,
                                   std::vector<const GenericDofMap*>& dof_maps,
                                   bool cells, bool interior_facets)
{
  const uint rank = dof_maps.size();

  // Get global dimensions
  std::vector<uint> global_dimensions(rank);
  for (uint i = 0; i < rank; ++i)
    global_dimensions[i] = dof_maps[i]->global_dimension();

  // Initialise sparsity pattern
  sparsity_pattern.init(global_dimensions);

  // Only build for rank >= 2 (matrices and higher order tensors)
  if (rank < 2)
    return;

  // Allocate some more space
  std::vector<uint> local_dimensions(rank);
  std::vector<uint> macro_local_dimensions(rank);
  uint** dofs = new uint*[rank];
  uint** macro_dofs = new uint*[rank];
  for (uint i = 0; i < rank; ++i)
  {
    local_dimensions[i] = dof_maps[i]->max_local_dimension();
    macro_local_dimensions[i] = 2*dof_maps[i]->max_local_dimension();

    dofs[i] = new uint[local_dimensions[i]];
    macro_dofs[i] = new uint[macro_local_dimensions[i]];
  }

  // Build sparsity pattern for cell integrals
  if (cells)
  {
    Progress p("Building sparsity pattern over cells", mesh.num_cells());
    UFCCell ufc_cell(mesh);

    for (CellIterator cell(mesh); !cell.end(); ++cell)
    {
      // Update to current cell
      ufc_cell.update(*cell);

      // Tabulate dofs for each dimension and get local dimensions
      for (uint i = 0; i < rank; ++i)
      {
        dof_maps[i]->tabulate_dofs(dofs[i], ufc_cell, cell->index());
        local_dimensions[i] = dof_maps[i]->local_dimension(ufc_cell);
      }

      // Fill sparsity pattern.
      sparsity_pattern.insert(&local_dimensions[0], dofs);
      p++;
    }
  }

  // FIXME: The below note is not true when there are no cell integrals,
  //        e.g. finite volume method
  // Note: no need to iterate over exterior facets since those dofs
  // are included when tabulating dofs on all cells

  // Build sparsity pattern for interior facet integrals
  if (interior_facets)
  {
    UFCCell ufc_cell0(mesh);
    UFCCell ufc_cell1(mesh);

    // Compute facets and facet - cell connectivity if not already computed
    mesh.init(mesh.topology().dim() - 1);
    mesh.init(mesh.topology().dim() - 1, mesh.topology().dim());
    if (!mesh.ordered())
      error("Mesh has not been ordered. Cannot compute sparsity pattern. Consider calling Mesh::order().");

    Progress p("Building sparsity pattern over interior facets", mesh.num_facets());

    for (FacetIterator facet(mesh); !facet.end(); ++facet)
    {
      // Check if we have an interior facet
      if (facet->num_entities(mesh.topology().dim()) != 2)
      {
        p++;
        continue;
      }

      // Get cells incident with facet
      Cell cell0(mesh, facet->entities(mesh.topology().dim())[0]);
      Cell cell1(mesh, facet->entities(mesh.topology().dim())[1]);

      // Update to current cell
      ufc_cell0.update(cell0);
      ufc_cell1.update(cell1);

      // Tabulate dofs for each dimension on macro element
      for (uint i = 0; i < rank; ++i)
      {
        const uint offset = dof_maps[i]->local_dimension(ufc_cell0);
        macro_local_dimensions[i] = offset + dof_maps[i]->local_dimension(ufc_cell1);
        dof_maps[i]->tabulate_dofs(macro_dofs[i], ufc_cell0, cell0.index());
        dof_maps[i]->tabulate_dofs(macro_dofs[i] + offset, ufc_cell1, cell1.index());
      }

      // Fill sparsity pattern.
      sparsity_pattern.insert(&macro_local_dimensions[0], macro_dofs);

      p++;
    }
  }

  // Finalize sparsity pattern
  sparsity_pattern.apply();

  // Clean up
  for (uint i = 0; i < rank; i++)
  {
    delete [] dofs[i];
    delete [] macro_dofs[i];
  }
  delete [] dofs;
  delete [] macro_dofs;
}
//-----------------------------------------------------------------------------
