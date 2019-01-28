// Copyright (C) 2007-2010 Garth N. Wells
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "SparsityPatternBuilder.h"
#include <algorithm>
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
    bool interior_facets, bool exterior_facets)
{
  // Get index maps
  assert(dofmaps[0]);
  assert(dofmaps[1]);
  std::array<std::shared_ptr<const common::IndexMap>, 2> index_maps
      = {{dofmaps[0]->index_map(), dofmaps[1]->index_map()}};


  const std::size_t D = mesh.topology().dim();

  // FIXME: Should check that index maps are matching

  // Create empty sparsity pattern
  la::SparsityPattern pattern(comm, index_maps);

  // Array to store macro-dofs, if required (for interior facets)
  std::array<Eigen::Array<PetscInt, Eigen::Dynamic, 1>, 2> macro_dofs;

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

  return pattern;
}
//-----------------------------------------------------------------------------
