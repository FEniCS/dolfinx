// Copyright (C) 2019 Chris Richardson
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include <dolfin/common/types.h>
#include <memory>
#include <ufc.h>
#include <vector>

#pragma once

struct ufc_dofmap;

namespace dolfin
{

namespace mesh
{
class CellType;
}

namespace fem
{

/// Element degree-of-freedom map

/// This class handles the mapping of degrees of freedom on an element.

class ElementDofMap
{
public:
  /// Constructor from UFC dofmap
  ElementDofMap(const ufc_dofmap& ufc_dofmap, const mesh::CellType& cell_type);

  // Copy constructor
  ElementDofMap(const ElementDofMap& dofmap) = delete;

  /// Move constructor
  ElementDofMap(ElementDofMap&& dofmap) = default;

  /// Destructor
  ~ElementDofMap() = default;

  ElementDofMap& operator=(const ElementDofMap& dofmap) = delete;

  /// Move assignment
  ElementDofMap& operator=(ElementDofMap&& dofmap) = default;

  /// Total number of dofs on element
  int num_dofs() const { return _num_dofs; }

  /// Number of dofs associated with entities of dimension dim
  int num_entity_dofs(int dim) const
  {
    assert(dim < 4);
    return _num_entity_dofs[dim];
  }

  /// Number of dofs associated with entities of dimension dim (plus
  /// connected entities of lower dim)
  int num_entity_closure_dofs(int dim) const
  {
    assert(dim < 4);
    return _num_entity_closure_dofs[dim];
  }

  /// Direct access to all entity dofs
  const std::vector<std::vector<std::vector<int>>>& entity_dofs() const
  {
    return _entity_dofs;
  }

  /// Direct access to all entity closure dofs
  const std::vector<std::vector<std::vector<int>>>& entity_closure_dofs() const
  {
    return _entity_closure_dofs;
  }

  /// Get number of sub-dofmaps
  int num_sub_dofmaps() const { return _sub_dofmaps.size(); }

  /// Get subdofmap given by list of components, one for each level.
  std::shared_ptr<const ElementDofMap>
  sub_dofmap(const std::vector<std::size_t>& component) const;

  /// Get mapping from a child dofmap, referenced by the component list
  /// (as for sub_dofmap()), back to this dofmap.
  std::vector<int>
  sub_dofmap_mapping(const std::vector<std::size_t>& component) const;

  // Block size
  int block_size() const { return _block_size; }

private:
  // Mapping of dofs to this ElementDofMap's immediate parent
  std::vector<int> _parent_map;

  // Block size, as deduced in from UFC
  int _block_size;

  // Total number of dofs in this element dofmap
  int _num_dofs;

  // The number of dofs associated with each entity type
  int _num_entity_dofs[4];

  // The number of dofs associated with each entity type, including
  // all connected entities of lower dimension.
  int _num_entity_closure_dofs[4];

  // List of dofs per entity, ordered by dimension.
  // dof = _entity_dofs[dim][entity][i]
  std::vector<std::vector<std::vector<int>>> _entity_dofs;

  // List of dofs with connected entities of lower dimension
  std::vector<std::vector<std::vector<int>>> _entity_closure_dofs;

  // List of sub dofmaps
  std::vector<std::shared_ptr<ElementDofMap>> _sub_dofmaps;
};
} // namespace fem
} // namespace dolfin
