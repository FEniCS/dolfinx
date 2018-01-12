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
// Modified by Anders Logg 2008-2014

#include <algorithm>

#include "SparsityPatternBuilder.h"
#include <dolfin/common/ArrayView.h>
#include <dolfin/common/MPI.h>
#include <dolfin/fem/GenericDofMap.h>
#include <dolfin/la/SparsityPattern.h>
#include <dolfin/log/log.h>
#include <dolfin/mesh/Cell.h>
#include <dolfin/mesh/Facet.h>
#include <dolfin/mesh/Mesh.h>
#include <dolfin/mesh/Vertex.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
void SparsityPatternBuilder::build(
    SparsityPattern& sparsity_pattern, const Mesh& mesh,
    const std::array<const GenericDofMap*, 2> dofmaps, bool cells,
    bool interior_facets, bool exterior_facets, bool vertices, bool diagonal,
    bool init, bool finalize)
{
  // Get index maps
  dolfin_assert(dofmaps[0]);
  dolfin_assert(dofmaps[1]);
  std::array<std::shared_ptr<const IndexMap>, 2> index_maps
      = {{dofmaps[0]->index_map(), dofmaps[1]->index_map()}};

  // FIXME: Should check that index maps are matching

  // Initialise sparsity pattern
  if (!sparsity_pattern.index_map(0) or !sparsity_pattern.index_map(1))
    throw std::runtime_error("SparsityPattern has not been initialised.");

  // Vector to store macro-dofs, if required (for interior facets)
  std::array<std::vector<dolfin::la_index_t>, 2> macro_dofs;

  // Create vector to point to dofs
  std::array<ArrayView<const dolfin::la_index_t>, 2> dofs;

  // Build sparsity pattern for reals (globally supported basis members)
  // NOTE: It is very important that this is done before other integrals
  //       so that insertion of global nodes is no-op below
  // NOTE: We assume that global dofs contribute a whole row which is
  //       memory suboptimal (for restricted Lagrange multipliers) but very
  //       fast and certainly much better than quadratic scaling of usual
  //       insertion below
  std::vector<std::size_t> global_dofs0;
  dofmaps[sparsity_pattern.primary_dim()]->tabulate_global_dofs(global_dofs0);
  sparsity_pattern.insert_full_rows_local(global_dofs0);

  // FIXME: We iterate over the entire mesh even if the function space
  // is restricted. This works out fine since the local dofmap
  // returned on each cell will be an empty vector, but we might think
  // about optimizing this further.

  // Build sparsity pattern for cell integrals
  if (cells)
  {
    for (CellIterator cell(mesh); !cell.end(); ++cell)
    {
      // Tabulate dofs for each dimension and get local dimensions
      for (std::size_t i = 0; i < 2; ++i)
      {
        auto dmap = dofmaps[i]->cell_dofs(cell->index());
        dofs[i].set(dmap.size(), dmap.data());
      }

      // Insert non-zeroes in sparsity pattern
      sparsity_pattern.insert_local(dofs);
    }
  }

  // Build sparsity pattern for vertex/point integrals
  const std::size_t D = mesh.topology().dim();
  if (vertices)
  {
    mesh.init(0);
    mesh.init(0, D);

    std::array<std::vector<dolfin::la_index_t>, 2> global_dofs;
    std::array<std::vector<std::size_t>, 2> local_to_local_dofs;

    // Resize local dof map vector
    for (std::size_t i = 0; i < 2; ++i)
    {
      global_dofs[i].resize(dofmaps[i]->num_entity_dofs(0));
      local_to_local_dofs[i].resize(dofmaps[i]->num_entity_dofs(0));
    }

    for (VertexIterator vert(mesh); !vert.end(); ++vert)
    {
      // Get mesh cell to which mesh vertex belongs (pick first)
      Cell mesh_cell(mesh, vert->entities(D)[0]);

      // Check that cell is not a ghost
      dolfin_assert(!mesh_cell.is_ghost());

      // Get local index of vertex with respect to the cell
      const std::size_t local_vertex = mesh_cell.index(*vert);
      for (std::size_t i = 0; i < 2; ++i)
      {
        auto dmap = dofmaps[i]->cell_dofs(mesh_cell.index());
        dofs[i].set(dmap.size(), dmap.data());
        dofmaps[i]->tabulate_entity_dofs(local_to_local_dofs[i], 0,
                                         local_vertex);

        // Copy cell dofs to local dofs and tabulated values to
        for (std::size_t j = 0; j < local_to_local_dofs[i].size(); ++j)
          global_dofs[i][j] = dofs[i][local_to_local_dofs[i][j]];
      }

      // Insert non-zeroes in sparsity pattern
      std::array<ArrayView<const dolfin::la_index_t>, 2> global_dofs_p;
      for (std::size_t i = 0; i < 2; ++i)
        global_dofs_p[i].set(global_dofs[i]);
      sparsity_pattern.insert_local(global_dofs_p);
    }
  }

  // Note: no need to iterate over exterior facets since those dofs
  //       are included when tabulating dofs on all cells

  // Build sparsity pattern for interior/exterior facet integrals
  if (interior_facets || exterior_facets)
  {
    // Compute facets and facet - cell connectivity if not already
    // computed
    mesh.init(D - 1);
    mesh.init(D - 1, D);
    if (!mesh.ordered())
    {
      dolfin_error(
          "SparsityPatternBuilder.cpp", "compute sparsity pattern",
          "Mesh is not ordered according to the UFC numbering convention. "
          "Consider calling mesh.order()");
    }

    for (FacetIterator facet(mesh); !facet.end(); ++facet)
    {
      bool this_exterior_facet = false;
      if (facet->num_global_entities(D) == 1)
        this_exterior_facet = true;

      // Check facet type
      if (exterior_facets && this_exterior_facet && !cells)
      {
        // Get cells incident with facet
        dolfin_assert(facet->num_entities(D) == 1);
        Cell cell(mesh, facet->entities(D)[0]);

        // Tabulate dofs for each dimension and get local dimensions
        for (std::size_t i = 0; i < 2; ++i)
        {
          auto dmap = dofmaps[i]->cell_dofs(cell.index());
          dofs[i].set(dmap.size(), dmap.data());
        }

        // Insert dofs
        sparsity_pattern.insert_local(dofs);
      }
      else if (interior_facets && !this_exterior_facet)
      {
        if (facet->num_entities(D) == 1)
        {
          dolfin_assert(facet->is_ghost());
          continue;
        }

        // Get cells incident with facet
        dolfin_assert(facet->num_entities(D) == 2);
        Cell cell0(mesh, facet->entities(D)[0]);
        Cell cell1(mesh, facet->entities(D)[1]);

        // Tabulate dofs for each dimension on macro element
        for (std::size_t i = 0; i < 2; i++)
        {
          // Get dofs for each cell
          auto cell_dofs0 = dofmaps[i]->cell_dofs(cell0.index());
          auto cell_dofs1 = dofmaps[i]->cell_dofs(cell1.index());

          // Create space in macro dof vector
          macro_dofs[i].resize(cell_dofs0.size() + cell_dofs1.size());

          // Copy cell dofs into macro dof vector
          std::copy(cell_dofs0.data(), cell_dofs0.data() + cell_dofs0.size(),
                    macro_dofs[i].begin());
          std::copy(cell_dofs1.data(), cell_dofs1.data() + cell_dofs1.size(),
                    macro_dofs[i].begin() + cell_dofs0.size());

          // Store pointer to macro dofs
          dofs[i].set(macro_dofs[i]);
        }

        // Insert dofs
        sparsity_pattern.insert_local(dofs);
      }
    }
  }

  if (diagonal)
  {
    const std::size_t primary_dim = sparsity_pattern.primary_dim();
    const std::size_t primary_codim = primary_dim == 0 ? 1 : 0;
    const auto primary_range = index_maps[primary_dim]->local_range();
    const std::size_t secondary_range
        = index_maps[primary_codim]->size(IndexMap::MapSize::GLOBAL);
    const std::size_t diagonal_range
        = std::min((std::size_t)primary_range[1], secondary_range);

    if (index_maps[0]->block_size() != index_maps[1]->block_size())
      throw std::runtime_error(
          "Add diagonal with non-matching block sizes not working yet.");
    std::size_t bs = index_maps[0]->block_size();

    std::vector<dolfin::la_index_t> indices(
        bs * (diagonal_range - primary_range[0]));
    std::iota(indices.begin(), indices.end(), bs * primary_range[0]);
    const std::array<ArrayView<const dolfin::la_index_t>, 2> diags = {
        {ArrayView<const dolfin::la_index_t>(indices.size(), indices.data()),
         ArrayView<const dolfin::la_index_t>(indices.size(), indices.data())}};

    sparsity_pattern.insert_global(diags);
  }

  // Finalize sparsity pattern (communicate off-process terms)
  if (finalize)
    sparsity_pattern.apply();
}
//-----------------------------------------------------------------------------
