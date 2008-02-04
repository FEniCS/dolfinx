// Copyright (C) 2007 Garth N. Wells.
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Ola Skavhaug
// Modified by Magnus Vikstrom 2008.
//
// First added:  2007-05-24
// Last changed: 2008-01-24

#include <dolfin/dolfin_log.h>
#include <dolfin/pDofMapSet.h>
#include <dolfin/Cell.h>
#include <dolfin/Facet.h>
#include <dolfin/Mesh.h>
#include <dolfin/GenericSparsityPattern.h>
#include <dolfin/pSparsityPatternBuilder.h>
#include <dolfin/pUFC.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
void pSparsityPatternBuilder::build(GenericSparsityPattern& sparsity_pattern, Mesh& mesh,
                                                pUFC& ufc, const pDofMapSet& dof_map_set)
{
  if (dof_map_set.size() == 0)
    scalarBuild(sparsity_pattern);
  else if (dof_map_set.size() == 1)
    vectorBuild(sparsity_pattern, dof_map_set);
  else if (dof_map_set.size() == 2)
    matrixBuild(sparsity_pattern, mesh, ufc, dof_map_set);
  else
    error("Cannot compute sparsity patterm for rank > 2.");
}
//-----------------------------------------------------------------------------
void pSparsityPatternBuilder::scalarBuild(GenericSparsityPattern& sparsity_pattern) 
{
  // Do nothing
}
//-----------------------------------------------------------------------------
void pSparsityPatternBuilder::vectorBuild(GenericSparsityPattern& sparsity_pattern, 
                                                    const pDofMapSet& dof_map_set)
{
  // Initialise sparsity pattern with problem size
  uint dims[1];
  dims[0] = dof_map_set[0].global_dimension();
  sparsity_pattern.init(1, dims);
}
//-----------------------------------------------------------------------------
void pSparsityPatternBuilder::matrixBuild(GenericSparsityPattern& sparsity_pattern, 
                                  Mesh& mesh, pUFC& ufc, const pDofMapSet& dof_map_set)
{
  // Initialise sparsity pattern
  uint dims[2];
  dims[0] = dof_map_set[0].global_dimension();
  dims[1] = dof_map_set[1].global_dimension();
  //sparsity_pattern.init(2, dims);
  sparsity_pattern.pinit(2, dims);

  // Create sparsity pattern for cell integrals
  if (ufc.form.num_cell_integrals() != 0)
  {
   // Build sparsity pattern
    uint num_rows[2];
    num_rows[0] = dof_map_set[0].local_dimension();
    num_rows[1] = dof_map_set[1].local_dimension();

    for (CellIterator cell(mesh); !cell.end(); ++cell)
    {
      // Update to current cell
      ufc.update(*cell);
  
      // Tabulate dofs for each dimension
      dof_map_set[0].tabulate_dofs(ufc.dofs[0], *cell);
      dof_map_set[1].tabulate_dofs(ufc.dofs[1], *cell);
 
      // Fill sparsity pattern.
      //sparsity_pattern.insert(num_rows, ufc.dofs);
      sparsity_pattern.pinsert(num_rows, ufc.dofs);
    }
  }

  // Create sparsity pattern for interior facet integrals
  if(ufc.form.num_interior_facet_integrals() != 0)
  {
    // Compute facets and facet - cell connectivity if not already computed
    mesh.init(mesh.topology().dim() - 1);
    mesh.init(mesh.topology().dim() - 1, mesh.topology().dim());
    mesh.order();

    uint num_rows[2];
    num_rows[0] = dof_map_set[0].macro_local_dimension();
    num_rows[1] = dof_map_set[1].macro_local_dimension();
  
    for (FacetIterator facet(mesh); !facet.end(); ++facet)
    {
      // Check if we have an interior facet
      if ( facet->numEntities(mesh.topology().dim()) != 2 )
        continue;

      // Get cells incident with facet
      Cell cell0(mesh, facet->entities(mesh.topology().dim())[0]);
      Cell cell1(mesh, facet->entities(mesh.topology().dim())[1]);

      // Update to current pair of cells
      ufc.update(cell0, cell1);
    
      // Tabulate dofs for each dimension on macro element
      for (uint i = 0; i < ufc.form.rank(); i++)
      {
        const uint offset = dof_map_set[i].local_dimension();
        dof_map_set[i].tabulate_dofs(ufc.macro_dofs[i], cell0);
        dof_map_set[i].tabulate_dofs(ufc.macro_dofs[i] + offset, cell1);
      }

      // Fill sparsity pattern.
      sparsity_pattern.insert(num_rows, ufc.macro_dofs);
    }
  }
  // Finalize sparsity pattern
  sparsity_pattern.apply();
}
//-----------------------------------------------------------------------------
