// Copyright (C) 2007-2026 Garth N. Wells
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "sparsitybuild.h"
#include "DofMap.h"
#include <algorithm>
#include <dolfinx/la/SparsityPattern.h>

using namespace dolfinx;
using namespace dolfinx::fem;

//-----------------------------------------------------------------------------
void sparsitybuild::interior_facets(
    la::SparsityPattern& pattern,
    std::array<std::span<const std::int32_t>, 2> cells,
    std::array<std::reference_wrapper<const DofMap>, 2> dofmaps)
{
  std::span<const std::int32_t> cells0 = cells[0];
  std::span<const std::int32_t> cells1 = cells[1];
  assert(cells0.size() == cells1.size());
  const DofMap& dofmap0 = dofmaps[0];
  const DofMap& dofmap1 = dofmaps[1];

  // Iterate over facets
  for (std::size_t f = 0; f < cells0.size(); f += 2)
  {
    // Test function dofs (sparsity pattern rows). A cell may not
    // exist on this side (e.g. an interface between two domains).
    std::span<const std::int32_t> dofs00
        = cells0[f] >= 0 ? dofmap0.cell_dofs(cells0[f])
                         : std::span<const std::int32_t>();
    std::span<const std::int32_t> dofs01
        = cells0[f + 1] >= 0 ? dofmap0.cell_dofs(cells0[f + 1])
                             : std::span<const std::int32_t>();

    // Trial function dofs (sparsity pattern columns)
    std::span<const std::int32_t> dofs10
        = cells1[f] >= 0 ? dofmap1.cell_dofs(cells1[f])
                         : std::span<const std::int32_t>();
    std::span<const std::int32_t> dofs11
        = cells1[f + 1] >= 0 ? dofmap1.cell_dofs(cells1[f + 1])
                             : std::span<const std::int32_t>();

    // Insert the four (test, trial) blocks directly, rather than via
    // a temporary buffer that could leak a previous facet's dofs
    pattern.insert(dofs00, dofs10);
    pattern.insert(dofs00, dofs11);
    pattern.insert(dofs01, dofs10);
    pattern.insert(dofs01, dofs11);
  }
}
//-----------------------------------------------------------------------------
