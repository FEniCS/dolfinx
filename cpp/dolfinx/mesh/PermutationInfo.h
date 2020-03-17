// Copyright (C) 2020 Matthew Scroggs
//
// This file is part of DOLFINX (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include <Eigen/Dense>
#include <array>
#include <bitset>
#include <cstdint>
#include <dolfinx/graph/AdjacencyList.h>
#include <dolfinx/graph/Partitioning.h>
#include <memory>
#include <vector>

// Define the size of the bitset
#define BITSETSIZE 32

namespace dolfinx
{

namespace mesh
{

class Topology;

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

  /// Get the permutation number to apply to a facet.
  /// The permutations are numbered so that:
  ///   n%2 gives the number of reflections to apply
  ///   n//2 gives the number of rotations to apply
  /// Each column of the returned array represents a cell, and each row a
  /// facet of that cell.
  /// @return The permutation number
  const Eigen::Array<std::uint8_t, Eigen::Dynamic, Eigen::Dynamic>&
  get_facet_permutations() const;

  /// Get the permutation number to apply to a facet.
  /// The permutations are numbered so that:
  ///   n%2 gives the number of reflections to apply
  ///   n//2 gives the number of rotations to apply
  /// Each column of the returned array represents a cell, and each row a
  /// facet of that cell.
  /// @return The permutation number
  const std::vector<std::uint32_t>& get_cell_data() const;

  /// Compute entity permutations and reflections used in assembly
  void create_entity_permutations(Topology& topology);

private:
  // The facet permutations
  Eigen::Array<std::uint8_t, Eigen::Dynamic, Eigen::Dynamic>
      _facet_permutations;

  // Cell data
  std::vector<std::uint32_t> _cell_data;
};

} // namespace mesh
} // namespace dolfinx
