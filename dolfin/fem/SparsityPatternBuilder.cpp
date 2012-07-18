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
                   const std::vector<const GenericDofMap*>& dofmaps,
                   const std::vector<std::pair<uint, uint> >& master_slave_dofs,
                   bool cells, bool interior_facets, bool exterior_facets)
{
  const uint rank = dofmaps.size();

  // Get global dimensions and local range
  std::vector<uint> global_dimensions(rank);
  std::vector<std::pair<uint, uint> > local_range(rank);
  std::vector<const boost::unordered_map<uint, uint>* > off_process_owner(rank);
  for (uint i = 0; i < rank; ++i)
  {
    global_dimensions[i] = dofmaps[i]->global_dimension();
    local_range[i]       = dofmaps[i]->ownership_range();
    off_process_owner[i] = &(dofmaps[i]->off_process_owner());
  }

  // Initialise sparsity pattern
  sparsity_pattern.init(global_dimensions, local_range, off_process_owner);

  // Only build for rank >= 2 (matrices and higher order tensors) that require
  // sparsity details
  if (rank < 2)
    return;

  // Create vector to point to dofs
  std::vector<const std::vector<uint>* > dofs(rank);

  // Build sparsity pattern for cell integrals
  if (cells)
  {
    Progress p("Building sparsity pattern over cells", mesh.num_cells());
    for (CellIterator cell(mesh); !cell.end(); ++cell)
    {
      // Tabulate dofs for each dimension and get local dimensions
      for (uint i = 0; i < rank; ++i)
        dofs[i] = &dofmaps[i]->cell_dofs(cell->index());

      // Insert non-zeroes in sparsity pattern
      sparsity_pattern.insert(dofs);
      p++;
    }
  }

  // Note: no need to iterate over exterior facets since those dofs
  //       are included when tabulating dofs on all cells

  // Build sparsity pattern for interior/exterior facet integrals
  const uint D = mesh.topology().dim();
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
    std::vector<std::vector<uint> > macro_dofs(rank);

    Progress p("Building sparsity pattern over interior facets", mesh.num_facets());
    for (FacetIterator facet(mesh); !facet.end(); ++facet)
    {
      const bool exterior_facet = facet->exterior();

      // Check facet type
      if (exterior_facets && exterior_facet && !cells)
      {
        // Get cells incident with facet
        dolfin_assert(facet->num_entities(D) == 1);
        Cell cell(mesh, facet->entities(D)[0]);

        // Tabulate dofs for each dimension and get local dimensions
        for (uint i = 0; i < rank; ++i)
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
        for (uint i = 0; i < rank; i++)
        {
          // Get dofs for each cell
          const std::vector<uint>& cell_dofs0 = dofmaps[i]->cell_dofs(cell0.index());
          const std::vector<uint>& cell_dofs1 = dofmaps[i]->cell_dofs(cell1.index());

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

  // Master-slave rows for periodic boundary conditions
  std::vector<uint> dofs0(1);
  std::vector<uint> dofs1;
  dofs[0] = &dofs0;
  dofs[1] = &dofs1;

  std::vector<std::pair<uint, uint> >::const_iterator master_slave;
  for (master_slave = master_slave_dofs.begin(); master_slave != master_slave_dofs.end(); ++master_slave)
  {
    const uint master_dof = master_slave->first;
    const uint slave_dof  = master_slave->second;

    if (master_dof >= local_range[0].first && master_dof < local_range[0].second)
    {
      // Get non-zero columns for master row
      sparsity_pattern.get_edges(master_dof, dofs1);
      dofs1.push_back(slave_dof);

      //cout << "Master, slave dof: " << master_dof << ", " << slave_dof << endl;
      //for (uint i = 0; i < dofs1.size(); ++i)
      //  cout << "  " << (*dofs[1])[i] << endl;

      // Add non-zero columns to slave row
      dofs0[0] = slave_dof;
      sparsity_pattern.insert(dofs);
    }

    if (slave_dof >= local_range[0].first && slave_dof < local_range[0].second)
    {
      // Get non-zero columns for slave row
      sparsity_pattern.get_edges(slave_dof, dofs1);
      dofs1.push_back(master_dof);

      //cout << "Slave, master dof: " << slave_dof << ", " << master_dof << endl;
      //for (uint i = 0; i < dofs1.size(); ++i)
      //  cout << "  " << (*dofs[1])[i] << endl;

      // Add non-zero columns to master row
      dofs0[0] = master_dof;
      sparsity_pattern.insert(dofs);
    }
  }

  // Finalize sparsity pattern
  sparsity_pattern.apply();
}
//-----------------------------------------------------------------------------
