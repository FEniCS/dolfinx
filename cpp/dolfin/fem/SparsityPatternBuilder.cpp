// Copyright (C) 2007-2010 Garth N. Wells
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "SparsityPatternBuilder.h"
#include <algorithm>
#include <dolfin/common/ArrayView.h>
#include <dolfin/common/IndexMap.h>
#include <dolfin/common/MPI.h>
#include <dolfin/fem/GenericDofMap.h>
#include <dolfin/la/SparsityPattern.h>
#include <dolfin/log/log.h>
#include <dolfin/mesh/Cell.h>
#include <dolfin/mesh/Facet.h>
#include <dolfin/mesh/Mesh.h>
#include <dolfin/mesh/MeshIterator.h>
#include <dolfin/mesh/Vertex.h>

using namespace dolfin;
using namespace dolfin::fem;

//-----------------------------------------------------------------------------
la::SparsityPattern SparsityPatternBuilder::build(
    MPI_Comm comm, const mesh::Mesh& mesh,
    const std::array<const fem::GenericDofMap*, 2> dofmaps, bool cells,
    bool interior_facets, bool exterior_facets, bool vertices, bool diagonal,
    bool finalize)
{
  // Get index maps
  assert(dofmaps[0]);
  assert(dofmaps[1]);
  std::array<std::shared_ptr<const common::IndexMap>, 2> index_maps
      = {{dofmaps[0]->index_map(), dofmaps[1]->index_map()}};

  // FIXME: Should check that index maps are matching

  // Create empty sparsity pattern
  la::SparsityPattern pattern(comm, index_maps);

  // Array to store macro-dofs, if required (for interior facets)
  std::array<EigenArrayXlaindex, 2> macro_dofs;

  // Create vector to point to dofs
  std::array<common::ArrayView<const dolfin::la_index_t>, 2> dofs;

  // Build sparsity pattern for reals (globally supported basis members)
  // NOTE: It is very important that this is done before other integrals
  //       so that insertion of global nodes is no-op below
  // NOTE: We assume that global dofs contribute a whole row which is
  //       memory suboptimal (for restricted Lagrange multipliers) but very
  //       fast and certainly much better than quadratic scaling of usual
  //       insertion below
  const Eigen::Array<std::size_t, Eigen::Dynamic, 1> global_dofs0
      = dofmaps[0]->tabulate_global_dofs();
  pattern.insert_full_rows_local(global_dofs0);

  // FIXME: We iterate over the entire mesh even if the function space
  // is restricted. This works out fine since the local dofmap
  // returned on each cell will be an empty vector, but we might think
  // about optimizing this further.

  // Build sparsity pattern for cell integrals
  if (cells)
  {
    for (auto& cell : mesh::MeshRange<mesh::Cell>(mesh))
    {
      pattern.insert_local(dofmaps[0]->cell_dofs(cell.index()),
                           dofmaps[1]->cell_dofs(cell.index()));
    }
  }

  // Build sparsity pattern for vertex/point integrals
  const std::size_t D = mesh.topology().dim();
  if (vertices)
  {
    throw std::runtime_error("Sparsity pattern building over vertices not working.");
    // mesh.init(0);
    // mesh.init(0, D);

    // std::array<std::vector<dolfin::la_index_t>, 2> global_dofs;
    // std::array<std::vector<int>, 2> local_to_local_dofs;

    // // Resize local dof map vector
    // for (std::size_t i = 0; i < 2; ++i)
    // {
    //   global_dofs[i].resize(dofmaps[i]->num_entity_dofs(0));
    //   local_to_local_dofs[i].resize(dofmaps[i]->num_entity_dofs(0));
    // }

    // for (auto& vert : mesh::MeshRange<mesh::Vertex>(mesh))
    // {
    //   // Get mesh cell to which mesh vertex belongs (pick first)
    //   mesh::Cell mesh_cell(mesh, vert.entities(D)[0]);

    //   // Check that cell is not a ghost
    //   assert(!mesh_cell.is_ghost());

    //   // Get local index of vertex with respect to the cell
    //   const std::size_t local_vertex = mesh_cell.index(vert);
    //   for (std::size_t i = 0; i < 2; ++i)
    //   {
    //     auto dmap = dofmaps[i]->cell_dofs(mesh_cell.index());
    //     dofs[i].set(dmap.size(), dmap.data());
    //     dofmaps[i]->tabulate_entity_dofs(local_to_local_dofs[i], 0,
    //                                      local_vertex);

    //     // Copy cell dofs to local dofs and tabulated values to
    //     for (std::size_t j = 0; j < local_to_local_dofs[i].size(); ++j)
    //       global_dofs[i][j] = dofs[i][local_to_local_dofs[i][j]];
    //   }

    //   // Insert non-zeroes in sparsity pattern
    //   std::array<common::ArrayView<const dolfin::la_index_t>, 2>
    //   global_dofs_p; for (std::size_t i = 0; i < 2; ++i)
    //     global_dofs_p[i].set(global_dofs[i]);
    //   pattern.insert_local(global_dofs_p);
    // }
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

    for (auto& facet : mesh::MeshRange<mesh::Facet>(mesh))
    {
      bool this_exterior_facet = false;
      if (facet.num_global_entities(D) == 1)
        this_exterior_facet = true;

      // Check facet type
      if (exterior_facets && this_exterior_facet && !cells)
      {
        // Get cells incident with facet
        assert(facet.num_entities(D) == 1);
        mesh::Cell cell(mesh, facet.entities(D)[0]);
        pattern.insert_local(dofmaps[0]->cell_dofs(cell.index()),
                             dofmaps[1]->cell_dofs(cell.index()));
      }
      else if (interior_facets && !this_exterior_facet)
      {
        if (facet.num_entities(D) == 1)
        {
          assert(facet.is_ghost());
          continue;
        }

        // Get cells incident with facet
        assert(facet.num_entities(D) == 2);
        mesh::Cell cell0(mesh, facet.entities(D)[0]);
        mesh::Cell cell1(mesh, facet.entities(D)[1]);

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
                    macro_dofs[i].data());
          std::copy(cell_dofs1.data(), cell_dofs1.data() + cell_dofs1.size(),
                    macro_dofs[i].data() + cell_dofs0.size());

          // Store pointer to macro dofs
          // dofs[i].set(macro_dofs[i]);
        }

        // Insert dofs
        pattern.insert_local(macro_dofs[0], macro_dofs[1]);
      }
    }
  }

  if (diagonal)
  {
    const auto primary_range = index_maps[0]->local_range();
    const std::size_t secondary_range
        = index_maps[1]->size(common::IndexMap::MapSize::GLOBAL);
    const std::size_t diagonal_range
        = std::min((std::size_t)primary_range[1], secondary_range);

    if (index_maps[0]->block_size() != index_maps[1]->block_size())
      throw std::runtime_error(
          "Add diagonal with non-matching block sizes not working yet.");
    std::size_t bs = index_maps[0]->block_size();

    std::vector<dolfin::la_index_t> indices(
        bs * (diagonal_range - primary_range[0]));
    std::iota(indices.begin(), indices.end(), bs * primary_range[0]);

    // const std::array<common::ArrayView<const dolfin::la_index_t>, 2> diags
    //     = {{common::ArrayView<const dolfin::la_index_t>(indices.size(),
    //                                                     indices.data()),
    //         common::ArrayView<const dolfin::la_index_t>(indices.size(),
    //                                                     indices.data())}};
    Eigen::Map<const EigenArrayXlaindex> rows(indices.data(), indices.size());
    pattern.insert_global(rows, rows);
  }

  // Finalize sparsity pattern (communicate off-process terms)
  if (finalize)
    pattern.apply();

  return pattern;
}
//-----------------------------------------------------------------------------
