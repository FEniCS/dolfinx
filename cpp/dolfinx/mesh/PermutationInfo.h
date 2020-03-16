// Copyright (C) 2020 Matthew Scroggs
//
// This file is part of DOLFINX (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include <Eigen/Dense>
#include <array>
#include <cstdint>
#include <memory>
#include <vector>

namespace dolfinx
{

namespace mesh
{

class Topology;

/// Compute the edge reflection array for consistent edge orientation
/// @param[in] topology The object topology
/// @return the Reflection array for each edge
Eigen::Array<bool, Eigen::Dynamic, Eigen::Dynamic>
compute_edge_reflections(const Topology& topology);

/// Compute the face reflection and rotation arrays for consistent face
/// orientation
/// @param[in] topology The object topology
/// @return the Reflection array for each face and the rotation array
///  for each face
std::pair<Eigen::Array<bool, Eigen::Dynamic, Eigen::Dynamic>,
          Eigen::Array<std::uint8_t, Eigen::Dynamic, Eigen::Dynamic>>
compute_face_permutations(const Topology& topology);

/// Information about how the entities of each cell should be permuted to get to a
/// low-to-high ordering
class PermutationInfo
{
public:
  /// Create empty mesh topology
  PermutationInfo();

  /// Copy constructor
  PermutationInfo(const PermutationInfo& info) = default;

  /// Move constructor
  PermutationInfo(PermutationInfo&& info) = default;

  /// Destructor
  ~PermutationInfo() = default;

  /// @todo Use std::vector<int32_t> to store 1/0 marker for each edge/face
  /// Get an array of bools that say whether each edge needs to be
  /// reflected to match the low->high ordering of the cell.
  /// Each column of the returned array represents a cell, and each row an
  /// edge of that cell.
  /// @return An Eigen::Array of bools
  const Eigen::Array<bool, Eigen::Dynamic, Eigen::Dynamic>&
  get_edge_reflections() const;

  /// @todo Use std::vector<int32_t> to store 1/0 marker for each edge/face
  /// Get an array of bools that say whether each face needs to be
  /// reflected to match the low->high ordering of the cell.
  /// Each column of the returned array represents a cell, and each row a
  /// face of that cell.
  /// @return An Eigen::Array of bools
  const Eigen::Array<bool, Eigen::Dynamic, Eigen::Dynamic>&
  get_face_reflections() const;

  /// Get an array of numbers that say how many times each face needs to be
  /// rotated to match the low->high ordering of the cell.
  /// Each column of the returned array represents a cell, and each row a
  /// face of that cell.
  /// @return An Eigen::Array of uint8_ts
  const Eigen::Array<std::uint8_t, Eigen::Dynamic, Eigen::Dynamic>&
  get_face_rotations() const;

  /// Get the permutation number to apply to a facet.
  /// The permutations are numbered so that:
  ///   n%2 gives the number of reflections to apply
  ///   n//2 gives the number of rotations to apply
  /// Each column of the returned array represents a cell, and each row a
  /// facet of that cell.
  /// @return The permutation number
  const Eigen::Array<std::uint8_t, Eigen::Dynamic, Eigen::Dynamic>&
  get_facet_permutations() const;

  /// Compute entity permutations and reflections used in assembly
  void create_entity_permutations(Topology& topology);

private:
  // TODO: Use std::vector<int32_t> to store 1/0 marker for each edge/face
  // The entity reflections of edges
  Eigen::Array<bool, Eigen::Dynamic, Eigen::Dynamic> _edge_reflections;

  // TODO: Use std::vector<int32_t> to store 1/0 marker for each edge/face
  // The entity reflections of faces
  Eigen::Array<bool, Eigen::Dynamic, Eigen::Dynamic> _face_reflections;

  // The entity reflections of faces
  Eigen::Array<std::uint8_t, Eigen::Dynamic, Eigen::Dynamic> _face_rotations;

  // The facet permutations
  Eigen::Array<std::uint8_t, Eigen::Dynamic, Eigen::Dynamic>
      _facet_permutations;
};

} // namespace mesh
} // namespace dolfinx
