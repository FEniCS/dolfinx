// Copyright (C) 2007-2010 Garth N. Wells
//
// This file is part of DOLFIN.
//
// DOLFIN is free software: you can redistribute it and/or modify
// it under the terms of the GNU Lesser General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// DOLFIN is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with DOLFIN. If not, see <http://www.gnu.org/licenses/>.
//
// Modified by Ola Skavhaug 2007
// Modified by Anders Logg 2008-2011
//
// First added:  2007-05-24
// Last changed: 2011-11-14

#include <dolfin/common/timing.h>
#include <dolfin/common/MPI.h>
#include <dolfin/la/GenericSparsityPattern.h>
#include <dolfin/mesh/Cell.h>
#include <dolfin/mesh/Facet.h>
#include <dolfin/mesh/Mesh.h>
#include "GenericDofMap.h"
#include "SparsityPatternBuilder.h"

#include <dolfin/log/dolfin_log.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
void SparsityPatternBuilder::build(GenericSparsityPattern& sparsity_pattern,
                   const Mesh& mesh,
                   const std::vector<const GenericDofMap*> dofmaps,
                   bool cells, bool interior_facets, bool exterior_facets, bool diagonal)
{
  const std::size_t rank = dofmaps.size();

  // Get global dimensions and local range
  std::vector<std::size_t> global_dimensions(rank);
  std::vector<std::pair<std::size_t, std::size_t> > local_range(rank);
  std::vector<const boost::unordered_map<std::size_t, std::size_t>* > off_process_owner(rank);
  for (std::size_t i = 0; i < rank; ++i)
  {
    global_dimensions[i] = dofmaps[i]->global_dimension();
    local_range[i]       = dofmaps[i]->ownership_range();
    off_process_owner[i] = &(dofmaps[i]->off_process_owner());
  }

  // Initialise sparsity pattern
  sparsity_pattern.init(global_dimensions, local_range, off_process_owner);

  // Only build for rank >= 2 (matrices and higher order tensors) that
  // require sparsity details
  if (rank < 2)
    return;

  // Create vector to point to dofs
  std::vector<const std::vector<dolfin::la_index>* > dofs(rank);

  // FIXME: We iterate over the entire mesh even if the function space
  // is restricted. This works out fine since the local dofmap
  // returned on each cell will be an empty vector, but we might think
  // about optimizing this further.

  // Build sparsity pattern for cell integrals
  if (cells)
  {
    Progress p("Building sparsity pattern over cells", mesh.num_cells());
    for (CellIterator cell(mesh); !cell.end(); ++cell)
    {
      // Tabulate dofs for each dimension and get local dimensions
      for (std::size_t i = 0; i < rank; ++i)
        dofs[i] = &dofmaps[i]->cell_dofs(cell->index());

      // Insert non-zeroes in sparsity pattern
      sparsity_pattern.insert(dofs);
      p++;
    }
  }

  // Note: no need to iterate over exterior facets since those dofs
  //       are included when tabulating dofs on all cells

  // Build sparsity pattern for interior/exterior facet integrals
  const std::size_t D = mesh.topology().dim();
  if (interior_facets || exterior_facets)
  {
    // Compute facets and facet - cell connectivity if not already computed
    mesh.init(D - 1);
    mesh.init(D - 1, D);
    if (!mesh.ordered())
    {
      dolfin_error("SparsityPatternBuilder.cpp",
                   "compute sparsity pattern",
                   "Mesh is not ordered according to the UFC numbering convention. "
                   "Consider calling mesh.order()");
    }

    // Vector to store macro-dofs (for interior facets)
    std::vector<std::vector<dolfin::la_index> > macro_dofs(rank);

    Progress p("Building sparsity pattern over interior facets", mesh.num_facets());
    for (FacetIterator facet(mesh); !facet.end(); ++facet)
    {
      bool exterior_facet = false;
      if (facet->num_global_entities(D) == 1)
        exterior_facet = true;

      // Check facet type
      if (exterior_facets && exterior_facet && !cells)
      {
        // Get cells incident with facet
        dolfin_assert(facet->num_entities(D) == 1);
        Cell cell(mesh, facet->entities(D)[0]);

        // Tabulate dofs for each dimension and get local dimensions
        for (std::size_t i = 0; i < rank; ++i)
          dofs[i] = &dofmaps[i]->cell_dofs(cell.index());

        // Insert dofs
        sparsity_pattern.insert(dofs);
      }
      else if (interior_facets && !exterior_facet)
      {
        // Get cells incident with facet
        Cell cell0(mesh, facet->entities(D)[0]);
        Cell cell1(mesh, facet->entities(D)[1]);

        // Tabulate dofs for each dimension on macro element
        for (std::size_t i = 0; i < rank; i++)
        {
          // Get dofs for each cell
          const std::vector<dolfin::la_index>& cell_dofs0 = dofmaps[i]->cell_dofs(cell0.index());
          const std::vector<dolfin::la_index>& cell_dofs1 = dofmaps[i]->cell_dofs(cell1.index());

          // Create space in macro dof vector
          macro_dofs[i].resize(cell_dofs0.size() + cell_dofs1.size());

          // Copy cell dofs into macro dof vector
          std::copy(cell_dofs0.begin(), cell_dofs0.end(), macro_dofs[i].begin());
          std::copy(cell_dofs1.begin(), cell_dofs1.end(), macro_dofs[i].begin() + cell_dofs0.size());

          // Store pointer to macro dofs
          dofs[i] = &macro_dofs[i];
        }

        // Insert dofs
        sparsity_pattern.insert(dofs);
      }

      p++;
    }
  }

  if (diagonal)
  {
    Progress p("Building sparsity pattern over diagonal", local_range[0].second-local_range[0].first);

    std::vector<dolfin::la_index> diagonal_dof(1, 0);
    for (std::size_t i = 0; i < rank; ++i)
      dofs[i] = &diagonal_dof;

    for (std::size_t j = local_range[0].first; j < local_range[0].second; j++)
    {
      diagonal_dof[0] = j;

      // Insert diagonal non-zeroes in sparsity pattern
      sparsity_pattern.insert(dofs);
      p++;
    }
  }

  // Finalize sparsity pattern (communicate off-process terms)
  sparsity_pattern.apply();
}
//-----------------------------------------------------------------------------
