// Copyright (C) 2007 Garth N. Wells.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2007-05-24
// Last changed:

#include <dolfin/dolfin_log.h>
#include <dolfin/Cell.h>
#include <dolfin/Facet.h>
#include <dolfin/Mesh.h>
#include <dolfin/SparsityPattern.h>
#include <dolfin/SparsityPatternBuilder.h>
#include <dolfin/UFC.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
void SparsityPatternBuilder::build(SparsityPattern& sparsity_pattern, Mesh& mesh,
                                   UFC& ufc)
{
  if (ufc.form.rank() == 0)
    scalarBuild(sparsity_pattern);
  else if (ufc.form.rank() == 1)
    vectorBuild(sparsity_pattern, ufc);
  else if (ufc.form.rank() == 2)
    matrixBuild(sparsity_pattern, mesh, ufc);
  else
    error("Cannot compute sparsity patterm for size > 2.");
}
//-----------------------------------------------------------------------------
void SparsityPatternBuilder::scalarBuild(SparsityPattern& sparsity_pattern) 
{
  // Do nothing
}
//-----------------------------------------------------------------------------
void SparsityPatternBuilder::vectorBuild(SparsityPattern& sparsity_pattern, 
                                         UFC& ufc)
{
  // Initialise sparsity pattern 
  sparsity_pattern.init(ufc.global_dimensions[0]);
}
//-----------------------------------------------------------------------------
void SparsityPatternBuilder::matrixBuild(SparsityPattern& sparsity_pattern, 
                                         Mesh& mesh, UFC& ufc)
{
  // Initialise sparsity pattern
  sparsity_pattern.init(ufc.global_dimensions[0], ufc.global_dimensions[1]);

  // Create sparsity pattern for cell integrals
  if (ufc.form.num_cell_integrals() != 0)
  {
    for (CellIterator cell(mesh); !cell.end(); ++cell)
    {
      // Update to current cell
      ufc.update(*cell);
  
      // Tabulate dofs for each dimension
      ufc.dof_maps[0]->tabulate_dofs(ufc.dofs[0], ufc.mesh, ufc.cell);
      ufc.dof_maps[1]->tabulate_dofs(ufc.dofs[1], ufc.mesh, ufc.cell);
  
      // Build sparsity pattern
      uint dim0 = ufc.dof_maps[0]->local_dimension();
      uint dim1 = ufc.dof_maps[1]->local_dimension();
      for (uint i = 0; i < dim0; ++i)
        for (uint j = 0; j < dim1; ++j)
          sparsity_pattern.insert( (ufc.dofs[0])[i], (ufc.dofs[0])[j] );
    }
  }

  // Create sparsity pattern for interior facet integrals
  if(ufc.form.num_interior_facet_integrals() != 0)
  {
    // Compute facets and facet - cell connectivity if not already computed
    mesh.init(mesh.topology().dim() - 1);
    mesh.init(mesh.topology().dim() - 1, mesh.topology().dim());
    mesh.order();
  
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
        const uint offset = ufc.local_dimensions[i];
        ufc.dof_maps[i]->tabulate_dofs(ufc.macro_dofs[i], ufc.mesh, ufc.cell0);
        ufc.dof_maps[i]->tabulate_dofs(ufc.macro_dofs[i] + offset, ufc.mesh, ufc.cell1);
      }

      // Build sparsity
      uint dim0 = ufc.macro_local_dimensions[0];
      uint dim1 = ufc.macro_local_dimensions[1];
      for (uint i = 0; i < dim0; ++i)
        for (uint j = 0; j < dim1; ++j)
          sparsity_pattern.insert( (ufc.macro_dofs[0])[i], (ufc.macro_dofs[1])[j] );
    }
  }
}
//-----------------------------------------------------------------------------
