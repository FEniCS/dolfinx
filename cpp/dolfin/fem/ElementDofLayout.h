// Copyright (C) 2019 Chris Richardson and Garth N. Wells
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include <array>
#include <dolfin/common/types.h>
#include <memory>
#include <set>
#include <vector>

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

class ElementDofLayout
{
public:
  /// Constructor
  ElementDofLayout(
      int block_size,
      const std::vector<std::vector<std::set<int>>>& entity_dofs,
      const std::vector<int>& parent_map,
      const std::vector<std::shared_ptr<const ElementDofLayout>> sub_dofmaps,
      const mesh::CellType& cell_type);

  // Copy-like constructor with option to reset (clear) parent map
  ElementDofLayout(const ElementDofLayout& element_dof_layout,
                   bool reset_parent);

  // Copy constructor
  ElementDofLayout(const ElementDofLayout& dofmap) = default;

  /// Move constructor
  ElementDofLayout(ElementDofLayout&& dofmap) = default;

  /// Destructor
  ~ElementDofLayout() = default;

  // Copy assignment
  ElementDofLayout& operator=(const ElementDofLayout& dofmap) = default;

  /// Move assignment
  ElementDofLayout& operator=(ElementDofLayout&& dofmap) = default;

  /// Number of dofs on element
  int num_dofs() const;

  /// Number of dofs associated with entities of dimension dim
  int num_entity_dofs(unsigned int dim) const;

  /// Number of dofs associated with entities of dimension dim (plus
  /// connected entities of lower dim)
  int num_entity_closure_dofs(unsigned int dim) const;

  /// Direct access to all entity dofs (dof = _entity_dofs[dim][entity][i])
  const std::vector<std::vector<std::set<int>>>& entity_dofs() const;

  /// Direct access to all entity closure dofs (dof =
  /// _entity_dofs[dim][entity][i])
  const std::vector<std::vector<std::set<int>>>& entity_closure_dofs() const;

  /// Get number of sub-dofmaps
  int num_sub_dofmaps() const;

  /// Get sub-dofmap given by list of components, one for each level
  std::shared_ptr<const ElementDofLayout>
  sub_dofmap(const std::vector<std::size_t>& component) const;

  /// Get view for a sub dofmap, defined by the component list (as for
  /// sub_dofmap()), into this dofmap. I.e., the dofs in this dofmap
  /// that are the sub-dofs.
  std::vector<int> sub_view(const std::vector<std::size_t>& component) const;

  /// Block size
  int block_size() const;

  /// Is view, i.e. has a parent dofmap
  bool is_view() const;

private:
  // Mapping of dofs to this ElementDofLayout's immediate parent
  std::vector<int> _parent_map;

  // Block size
  const int _block_size;

  // Total number of dofs on this element dofmap
  int _num_dofs;

  // The number of dofs associated with each entity type
  std::array<int, 4> _num_entity_dofs;

  // The number of dofs associated with each entity type, including all
  // connected entities of lower dimension.
  std::array<int, 4> _num_entity_closure_dofs;

  // List of dofs per entity, ordered by dimension.
  // dof = _entity_dofs[dim][entity][i]
  const std::vector<std::vector<std::set<int>>> _entity_dofs;

  // List of dofs with connected entities of lower dimension
  std::vector<std::vector<std::set<int>>> _entity_closure_dofs;

  // List of sub dofmaps
  const std::vector<std::shared_ptr<const ElementDofLayout>> _sub_dofmaps;
};
} // namespace fem
} // namespace dolfin
