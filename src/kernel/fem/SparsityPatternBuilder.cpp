// Copyright (C) 2007 Garth N. Wells.
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Ola Skavhaug
//
// First added:  2007-05-24
// Last changed: 2007-12-07

#include <dolfin/dolfin_log.h>
#include <dolfin/DofMapSet.h>
#include <dolfin/Cell.h>
#include <dolfin/Facet.h>
#include <dolfin/Mesh.h>
#include <dolfin/GenericSparsityPattern.h>
#include <dolfin/SparsityPatternBuilder.h>
#include <dolfin/UFC.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
void SparsityPatternBuilder::build(GenericSparsityPattern& sparsity_pattern, Mesh& mesh,
                                                UFC& ufc, const DofMapSet& dof_map_set)
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
void SparsityPatternBuilder::scalarBuild(GenericSparsityPattern& sparsity_pattern) 
{
  // Do nothing
}
//-----------------------------------------------------------------------------
void SparsityPatternBuilder::vectorBuild(GenericSparsityPattern& sparsity_pattern, 
                                                    const DofMapSet& dof_map_set)
{
  // Initialise sparsity pattern with problem size
  uint dims[1];
  dims[0] = dof_map_set[0].global_dimension();
  sparsity_pattern.init(1, dims);
}
//-----------------------------------------------------------------------------
void SparsityPatternBuilder::matrixBuild(GenericSparsityPattern& sparsity_pattern, 
                                  Mesh& mesh, UFC& ufc, const DofMapSet& dof_map_set)
{
  // Initialise sparsity pattern
  uint dims[2];
  dims[0] = dof_map_set[0].global_dimension();
  dims[1] = dof_map_set[1].global_dimension();
  sparsity_pattern.init(2, dims);

  // Create sparsity pattern for cell integrals
  if (ufc.form.num_cell_integrals() != 0)
  {
    for (CellIterator cell(mesh); !cell.end(); ++cell)
    {
      // Update to current cell
      ufc.update(*cell);
  
      // Tabulate dofs for each dimension
      dof_map_set[0].tabulate_dofs(ufc.dofs[0], *cell);
      dof_map_set[1].tabulate_dofs(ufc.dofs[1], *cell);
 
     // Build sparsity pattern
      uint num_rows[2];
      uint dim0 = dof_map_set[0].local_dimension();
      uint dim1 = dof_map_set[1].local_dimension();
      num_rows[0] = dim0;
      num_rows[1] = dim1;
      uint* rows[2];
      rows[0] = (uint*)malloc(dim0*sizeof(uint));
      rows[1] = (uint*)malloc(dim1*sizeof(uint));
      for (uint i = 0; i < dim0; ++i)
        rows[0][i] = ufc.dofs[0][i];
      for (uint j = 0; j < dim1; ++j)
        rows[1][j] = ufc.dofs[0][j];
      sparsity_pattern.insert(num_rows, rows);
      free(rows[0]);
      free(rows[1]);
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
        const uint offset = dof_map_set[i].local_dimension();
        dof_map_set[i].tabulate_dofs(ufc.macro_dofs[i], cell0);
        dof_map_set[i].tabulate_dofs(ufc.macro_dofs[i] + offset, cell1);
      }

      // Build sparsity
      // FIXME: local macro dimension should come from DofMap
      uint num_rows[2];
      uint dim0 = dof_map_set[0].macro_local_dimension();
      uint dim1 = dof_map_set[1].macro_local_dimension();
      num_rows[0] = dim0;
      num_rows[1] = dim1;
      uint* rows[2];
      rows[0] = (uint*)malloc(dim0*sizeof(uint));
      rows[1] = (uint*)malloc(dim1*sizeof(uint));
      for (uint i = 0; i < dim0; ++i)
        rows[0][i] = ufc.macro_dofs[0][i];
      for (uint j = 0; j < dim1; ++j)
        rows[1][j] = ufc.macro_dofs[0][j];
      sparsity_pattern.insert(num_rows, rows);
      free(rows[0]);
      free(rows[1]);
    }
  }
  // Finalize sparsity pattern
  sparsity_pattern.apply();
}
//-----------------------------------------------------------------------------
