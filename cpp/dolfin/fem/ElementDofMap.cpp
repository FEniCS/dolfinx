// Copyright (C) 2019 Chris Richardson
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "ElementDofMap.h"
#include <cstdlib>

using namespace dolfin;
using namespace dolfin::fem;

//-----------------------------------------------------------------------------
ElementDofMap::ElementDofMap(const ufc_dofmap& dofmap)
{
  _cell_dimension
      = dofmap.num_element_support_dofs + dofmap.num_global_support_dofs;

  // Copy over number of dofs per entity
  std::copy(dofmap.num_entity_dofs, dofmap.num_entity_dofs + 4,
            _num_entity_dofs);

  // Fill entity dof indices
  for (unsigned int dim = 0; dim < 4; ++dim)
  {
    _entity_dofs[dim].resize(_num_entity_dofs[dim]);
  }

  if (dofmap.num_sub_dofmaps > 0)
  {
    ufc_dofmap* sub_dofmap = dofmap.create_sub_dofmap(0);
    sub_dofmaps.push_back(std::make_unique<ElementDofMap>(*sub_dofmap));
    std::free(sub_dofmap);
  }
}
//-----------------------------------------------------------------------------
std::vector<int> ElementDofMap::tabulate_entity_dofs(int dim, int i) const
{
  return _entity_dofs[dim][i];
}
