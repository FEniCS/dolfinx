// Copyright (C) 2007-2008 Garth N. Wells.
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Ola Skavhaug, 2007.
// Modified by Anders Logg, 2008.
//
// First added:  2007-05-24
// Last changed: 2008-02-15

#include <dolfin/mesh/Cell.h>
#include <dolfin/mesh/Facet.h>
#include <dolfin/mesh/Mesh.h>
#include <dolfin/la/GenericSparsityPattern.h>
#include <dolfin/function/FunctionSpace.h>
#include "SparsityPatternBuilder.h"
#include "DofMap.h"
#include "UFC.h"
#include "Form.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
void SparsityPatternBuilder::build(GenericSparsityPattern& sparsity_pattern,
				   const Mesh& mesh, UFC& ufc, const std::vector<const DofMap*> dof_maps)
{
  // Initialise sparsity pattern
  /*
  if (dof_map_set.parallel())
    sparsity_pattern.pinit(ufc.form.rank(), ufc.global_dimensions);
  else
    sparsity_pattern.init(ufc.form.rank(), ufc.global_dimensions);
  */
  sparsity_pattern.init(ufc.form.rank(), ufc.global_dimensions);

  // Only build for rank >= 2 (matrices and higher order tensors)
  if (ufc.form.rank() < 2)
    return;

  // Build sparsity pattern for cell integrals
  if (ufc.form.num_cell_integrals() != 0)
  {
    for (CellIterator cell(mesh); !cell.end(); ++cell)
    {
      // Update to current cell
      ufc.update(*cell);

      // Tabulate dofs for each dimension
      for (uint i = 0; i < ufc.form.rank(); ++i)
        dof_maps[i]->tabulate_dofs(ufc.dofs[i], ufc.cell, cell->index());

    std::cout <<"--- ufc.dofs[0][0] "<<ufc.dofs[0][0]<<std::endl;
    std::cout <<"--- ufc.dofs[0][1] "<<ufc.dofs[0][1]<<std::endl;
    std::cout <<"--- ufc.dofs[0][2] "<<ufc.dofs[0][2]<<std::endl;
    std::cout <<"--- ufc.dofs[1][0] "<<ufc.dofs[1][0]<<std::endl;
    std::cout <<"--- ufc.dofs[1][1] "<<ufc.dofs[1][1]<<std::endl;
    std::cout <<"--- ufc.dofs[1][2] "<<ufc.dofs[1][2]<<std::endl;


      // Fill sparsity pattern.
      /*
      if( dof_map_set.parallel() )
        sparsity_pattern.pinsert(ufc.local_dimensions, ufc.dofs);
      else
        sparsity_pattern.insert(ufc.local_dimensions, ufc.dofs);
      */
      sparsity_pattern.insert(ufc.local_dimensions, ufc.dofs);
    }
  }

  // FIXME: The below note is not true when there are no cell integrals, e.g. finite volume method
  // Note: no need to iterate over exterior facets since those dofs
  // are included when tabulating dofs on all cells

  // Build sparsity pattern for interior facet integrals
  if (ufc.form.num_interior_facet_integrals() != 0)
  {
    // Compute facets and facet - cell connectivity if not already computed
    mesh.init(mesh.topology().dim() - 1);
    mesh.init(mesh.topology().dim() - 1, mesh.topology().dim());
    const_cast<Mesh&>(mesh).order();

    for (FacetIterator facet(mesh); !facet.end(); ++facet)
    {
      // Check if we have an interior facet
      if (facet->num_entities(mesh.topology().dim()) != 2)
        continue;

      // Get cells incident with facet
      Cell cell0(mesh, facet->entities(mesh.topology().dim())[0]);
      Cell cell1(mesh, facet->entities(mesh.topology().dim())[1]);

      // Update to current pair of cells
      ufc.update(cell0, cell1);

      // Tabulate dofs for each dimension on macro element
      for (uint i = 0; i < ufc.form.rank(); ++i)
      {
        const uint offset = dof_maps[i]->local_dimension(ufc.cell0);
        dof_maps[i]->tabulate_dofs(ufc.macro_dofs[i], ufc.cell0, cell0.index());
        dof_maps[i]->tabulate_dofs(ufc.macro_dofs[i] + offset, ufc.cell1, cell1.index());
      }

      // Fill sparsity pattern.
      sparsity_pattern.insert(ufc.macro_local_dimensions, ufc.macro_dofs);
    }
  }

  // Finalize sparsity pattern
  sparsity_pattern.apply();
}
//-----------------------------------------------------------------------------
void SparsityPatternBuilder::build(GenericSparsityPattern& sparsity_pattern,
				   UFC& ufc, const Form& a)
{

  const Mesh& mesh = a.mesh();
  // Initialise sparsity pattern
  /*
  if (dof_map_set.parallel())
    sparsity_pattern.pinit(ufc.form.rank(), ufc.global_dimensions);
  else
    sparsity_pattern.init(ufc.form.rank(), ufc.global_dimensions);
  */
  sparsity_pattern.init(ufc.form.rank(), ufc.global_dimensions);

  // Only build for rank >= 2 (matrices and higher order tensors)
  if (ufc.form.rank() < 2)
    return;

  uint num_function_spaces = a.function_spaces().size();
  // Build sparsity pattern for cell integrals
  if (ufc.form.num_cell_integrals() != 0)
  {
    for (CellIterator cell(mesh); !cell.end(); ++cell)
    {

      // FIXME will move this check into a separate function.
      // Need to check the coefficients as well.
      bool compute_on_cell = true;
      for (uint i = 0; i < num_function_spaces; i++)
      {
        if (!a.function_space(i).is_inside_restriction(cell->index()))
          compute_on_cell = false;
      }
      if (!compute_on_cell)
        continue;

      // Update to current cell
      ufc.update(*cell);

      // Tabulate dofs for each dimension
      for (uint i = 0; i < ufc.form.rank(); ++i)
        a.function_space(i).dofmap().tabulate_dofs(ufc.dofs[i], ufc.cell, cell->index());

      // Fill sparsity pattern.
      /*
      if( dof_map_set.parallel() )
        sparsity_pattern.pinsert(ufc.local_dimensions, ufc.dofs);
      else
        sparsity_pattern.insert(ufc.local_dimensions, ufc.dofs);
      */
      sparsity_pattern.insert(ufc.local_dimensions, ufc.dofs);
    }
  }

  // FIXME: The below note is not true when there are no cell integrals, e.g. finite volume method
  // Note: no need to iterate over exterior facets since those dofs
  // are included when tabulating dofs on all cells

  // Build sparsity pattern for interior facet integrals
  if (ufc.form.num_interior_facet_integrals() != 0)
  {
    // Compute facets and facet - cell connectivity if not already computed
    mesh.init(mesh.topology().dim() - 1);
    mesh.init(mesh.topology().dim() - 1, mesh.topology().dim());
    const_cast<Mesh&>(mesh).order();

    for (FacetIterator facet(mesh); !facet.end(); ++facet)
    {
      // Check if we have an interior facet
      if (facet->num_entities(mesh.topology().dim()) != 2)
        continue;

      // Get cells incident with facet
      Cell cell0(mesh, facet->entities(mesh.topology().dim())[0]);
      Cell cell1(mesh, facet->entities(mesh.topology().dim())[1]);

      // Update to current pair of cells
      ufc.update(cell0, cell1);

      // Tabulate dofs for each dimension on macro element
      for (uint i = 0; i < ufc.form.rank(); ++i)
      {
        const uint offset = a.function_space(i).dofmap().local_dimension(ufc.cell0);
        a.function_space(i).dofmap().tabulate_dofs(ufc.macro_dofs[i], ufc.cell0, cell0.index());
        a.function_space(i).dofmap().tabulate_dofs(ufc.macro_dofs[i] + offset, ufc.cell1, cell1.index());
      }

      // Fill sparsity pattern.
      sparsity_pattern.insert(ufc.macro_local_dimensions, ufc.macro_dofs);
    }
  }

  // Finalize sparsity pattern
  sparsity_pattern.apply();
}

