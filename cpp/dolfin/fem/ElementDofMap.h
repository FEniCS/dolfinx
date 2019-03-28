// Copyright (C) 2019 Chris Richardson
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include <array>
#include <dolfin/common/types.h>
#include <memory>
#include <vector>

struct ufc_dofmap;

namespace dolfin
{

namespace mesh
{
class CellType;
}

namespace fem
{
/// The class represents the degree-of-freedom (dofs) for an element.
/// Dofs are associated with a mesh entity. This class also handles
/// sub-space dofs, which are views into the parent dofs.

// TODO: For this class/concept to be robust, the topology of the
//       reference cell needs to be defined.
// TODO: Handle block dofmaps properly

class ElementDofMap
{
public:
  /// Constructor from UFC dofmap
  ElementDofMap(const ufc_dofmap& dofmap, const mesh::CellType& cell_type);

  ElementDofMap(int block_size, int num_dofs,
                std::array<int, 4> num_entity_dofs,
                std::array<int, 4> num_entity_closure_dofs,
                std::vector<std::vector<std::vector<int>>> entity_dofs,
                std::vector<std::vector<std::vector<int>>> entity_closure_dofs,
                std::vector<std::shared_ptr<ElementDofMap>> sub_dofmaps)
      : _block_size(block_size), _num_dofs(num_dofs),
        _num_entity_dofs(num_entity_dofs),
        _num_entity_closure_dofs(num_entity_closure_dofs),
        _entity_dofs(entity_dofs), _entity_closure_dofs(entity_closure_dofs),
        _sub_dofmaps(sub_dofmaps)
  {
    // Do nothing
  }

  // Copy constructor
  ElementDofMap(const ElementDofMap& dofmap) = delete;

  /// Move constructor
  ElementDofMap(ElementDofMap&& dofmap) = default;

  /// Destructor
  ~ElementDofMap() = default;

  ElementDofMap& operator=(const ElementDofMap& dofmap) = delete;

  /// Move assignment
  ElementDofMap& operator=(ElementDofMap&& dofmap) = default;

  /// Number of dofs on element
  int num_dofs() const;

  /// Number of dofs associated with entities of dimension dim
  int num_entity_dofs(int dim) const;

  /// Number of dofs associated with entities of dimension dim (plus
  /// connected entities of lower dim)
  int num_entity_closure_dofs(int dim) const;

  /// Direct access to all entity dofs
  const std::vector<std::vector<std::vector<int>>>& entity_dofs() const;

  /// Direct access to all entity closure dofs
  const std::vector<std::vector<std::vector<int>>>& entity_closure_dofs() const;

  /// Get number of sub-dofmaps
  int num_sub_dofmaps() const;

  /// Get sub-dofmap given by list of components, one for each level
  std::shared_ptr<const ElementDofMap>
  sub_dofmap(const std::vector<std::size_t>& component) const;

  /// Get mapping from a child dofmap, referenced by the component list
  /// (as for sub_dofmap()), back to this dofmap
  std::vector<int>
  sub_dofmap_mapping(const std::vector<std::size_t>& component) const;

  // Block size
  int block_size() const;

private:
public:
  // Mapping of dofs to this ElementDofMap's immediate parent
  std::vector<int> _parent_map;

private:
  // Block size, as deduced in from UFC
  int _block_size;

  // Total number of dofs in this element dofmap
  int _num_dofs;

  // The number of dofs associated with each entity type
  std::array<int, 4> _num_entity_dofs;

  // The number of dofs associated with each entity type, including
  // all connected entities of lower dimension.
  std::array<int, 4> _num_entity_closure_dofs;

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
