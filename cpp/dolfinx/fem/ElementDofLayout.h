// Copyright (C) 2019 Chris Richardson and Garth N. Wells
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include <array>
#include <span>
#include <vector>

namespace dolfinx::mesh
{
enum class CellType;
}

namespace dolfinx::fem
{

// TODO: For this class/concept to be robust, the topology of the
//       reference cell needs to be defined.
//
// TODO: Handle block dofmaps properly

/// Representation of the degree-of-freedom (dofs) for an element. Dofs
/// are associated with a mesh entity. This class also handles sub-space
/// dofs, which are views into the parent dofs.
class ElementDofLayout
{
public:
  /// @brief Constructor
  ///
  /// @param[in] block_size Number of times each degree-of-freedom is
  /// 'replicated', e.g. number of DOFs 'co-located' at each 'point'.
  /// @param[in] entity_dofs Dofs on each entity, in the format
  /// `entity_dofs[entity_dim][entity_number]=[dof0, dof1, ...]`.
  /// @param[in] entity_closure_dofs Dofs on the closure of each entity,
  /// in the format
  /// `entity_closure_dofs[entity_dim][entity_number]=[dof0, dof1,
  /// ...]`.
  /// @param[in] parent_map TODO
  /// @param[in] sub_layouts TODO
  ElementDofLayout(
      int block_size,
      const std::vector<std::vector<std::vector<int>>>& entity_dofs,
      const std::vector<std::vector<std::vector<int>>>& entity_closure_dofs,
      const std::vector<int>& parent_map,
      const std::vector<ElementDofLayout>& sub_layouts);

  /// @brief Copy the DOF layout, discarding any parent information.
  ///
  /// @return Copy of the layout.
  ElementDofLayout copy() const;

  /// @brief Copy constructor
  ElementDofLayout(const ElementDofLayout& dofmap) = default;

  /// @brief Move constructor
  ElementDofLayout(ElementDofLayout&& dofmap) = default;

  /// @brief Destructor
  ~ElementDofLayout() = default;

  /// @brief Copy assignment
  ElementDofLayout& operator=(const ElementDofLayout& dofmap) = default;

  /// @brief Move assignment
  ElementDofLayout& operator=(ElementDofLayout&& dofmap) = default;

  /// @brief Equality operator
  ///
  /// Sub- and parent dof layout data is not compared. The block sizes
  /// of the layouts are not compared.
  ///
  /// @return True if the layout data is the same, false otherwise.
  bool operator==(const ElementDofLayout& layout) const;

  /// @brief Dimension of the local finite element function space on a
  /// cell (number of dofs on element).
  /// @return Dimension of the local finite element function space.
  int num_dofs() const;

  /// @brief Return the number of dofs for a given entity dimension.
  /// @param[in] dim Entity dimension.
  /// @return Number of dofs associated with given entity dimension.
  int num_entity_dofs(int dim) const;

  /// @brief Return the number of closure dofs for a given entity
  /// dimension.
  /// @param[in] dim Entity dimension.
  /// @return Number of dofs associated with closure of given entity
  /// dimension.
  int num_entity_closure_dofs(int dim) const;

  /// @brief Local-local mapping of dofs on entity of cell.
  /// @param[in] dim Entity dimension.
  /// @param[in] entity_index Local entity index on the cell.
  /// @return Cell-local degree-of-freedom indices.
  const std::vector<int>& entity_dofs(int dim, int entity_index) const;

  /// @brief Local-local closure dofs on entity of cell.
  /// @param[in] dim Entity dimension.
  /// @param[in] entity_index Local entity index on the cell.
  /// @return Cell-local degree-of-freedom indices.
  const std::vector<int>& entity_closure_dofs(int dim, int entity_index) const;

  /// @brief Direct access to all entity dofs.
  ///
  /// Storage is `dof=_entity_dofs[dim][entity][i]).
  /// @return Entity dofs data structure.
  const std::vector<std::vector<std::vector<int>>>& entity_dofs_all() const;

  /// @brief Direct access to all entity closure dofs.
  ///
  /// Storage is  `dof=_entity_dofs[dim][entity][i])`.
  /// @return Entity dofs data structure for closure dofs.
  const std::vector<std::vector<std::vector<int>>>&
  entity_closure_dofs_all() const;

  /// @brief  Get number of sub-layouts.
  /// @return Number of sub-maps.
  int num_sub_dofmaps() const;

  /// @brief Get sub-dofmap given by list of components, one for each
  /// level.
  const ElementDofLayout& sub_layout(std::span<const int> component) const;

  /// @brief Get view for a sub-layout, defined by the component list
  /// (as for ::sub_layout), into this dofmap. I.e., the dofs in this
  /// dofmap that are the sub-dofs.
  std::vector<int> sub_view(std::span<const int> component) const;

  /// @brief Block size.
  int block_size() const;

  /// @brief True iff dof map is a view into another map.
  /// @returns bool True if the dof map is a sub-dof map (a view into
  /// another map).
  bool is_view() const;

private:
  // Block size
  int _block_size;

  // Mapping of dofs to this ElementDofLayout's immediate parent
  std::vector<int> _parent_map;

  // Total number of dofs on this element dofmap
  int _num_dofs;

  // The number of dofs associated with each entity type
  std::array<int, 4> _num_entity_dofs;

  // The number of dofs associated with each entity type, including all
  // connected entities of lower dimension.
  std::array<int, 4> _num_entity_closure_dofs;

  // List of dofs per entity, ordered by dimension.
  // dof = _entity_dofs[dim][entity][i]
  std::vector<std::vector<std::vector<int>>> _entity_dofs;

  // List of dofs with connected entities of lower dimension
  std::vector<std::vector<std::vector<int>>> _entity_closure_dofs;

  // List of sub dofmaps
  // std::vector<std::shared_ptr<const ElementDofLayout>> _sub_dofmaps;
  std::vector<ElementDofLayout> _sub_dofmaps;
};

} // namespace dolfinx::fem
