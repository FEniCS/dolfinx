// Copyright (C) 2007-2008 Garth N. Wells.
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Ola Skavhaug, 2007.
// Modified by Anders Logg, 2008-2009.
//
// First added:  2007-05-24
// Last changed: 2009-05-19

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
                                   const Form& a,
                                   const UFC& ufc)
{
  const Mesh& mesh = a.mesh();

  // Initialise sparsity pattern
  sparsity_pattern.init(ufc.form.rank(), ufc.global_dimensions);

  // Only build for rank >= 2 (matrices and higher order tensors)
  if (ufc.form.rank() < 2)
    return;

  // Build sparsity pattern for cell integrals
  if (ufc.form.num_cell_integrals() != 0)
  {
    UFCCell ufc_cell(mesh);

    for (CellIterator cell(mesh); !cell.end(); ++cell)
    {
      // Update to current cell
      ufc_cell.update(*cell);

      // FIXME: Use new restricted iterators for this (avoid continue)
      bool compute_on_cell = true;
      for (uint i = 0; i < a.function_spaces().size(); i++)
      {
        if (!a.function_space(i).is_inside_restriction(cell->index()))
          compute_on_cell = false;
      }
      if (!compute_on_cell)
        continue;

      // Tabulate dofs for each dimension and get local dimensions
      for (uint i = 0; i < ufc.form.rank(); ++i)
      {
        a.function_space(i).dofmap().tabulate_dofs(ufc.dofs[i], ufc_cell, cell->index());
        ufc.local_dimensions[i] = a.function_space(i).dofmap().local_dimension(ufc_cell);
      }

      // Fill sparsity pattern.
      sparsity_pattern.insert(ufc.local_dimensions, ufc.dofs);
    }
  }

  // FIXME: The below note is not true when there are no cell integrals, 
  //        e.g. finite volume method
  // Note: no need to iterate over exterior facets since those dofs
  // are included when tabulating dofs on all cells

  // Build sparsity pattern for interior facet integrals
  if (ufc.form.num_interior_facet_integrals() != 0)
  {
    UFCCell ufc_cell0(mesh);
    UFCCell ufc_cell1(mesh);

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

      // Update to current cell
      ufc_cell0.update(cell0);
      ufc_cell1.update(cell1);

      // Tabulate dofs for each dimension on macro element
      for (uint i = 0; i < ufc.form.rank(); ++i)
      {
        const uint offset = a.function_space(i).dofmap().local_dimension(ufc_cell0);
        ufc.macro_local_dimensions[i] = offset + a.function_space(i).dofmap().local_dimension(ufc_cell1);
        a.function_space(i).dofmap().tabulate_dofs(ufc.macro_dofs[i], ufc_cell0, cell0.index());
        a.function_space(i).dofmap().tabulate_dofs(ufc.macro_dofs[i] + offset, ufc_cell1, cell1.index());
      }

      // Fill sparsity pattern.
      sparsity_pattern.insert(ufc.macro_local_dimensions, ufc.macro_dofs);
    }
  }

  // Finalize sparsity pattern
  sparsity_pattern.apply();
}
//-----------------------------------------------------------------------------
