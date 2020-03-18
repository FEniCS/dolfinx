// Copyright (C) 2020 Matthew Scroggs
//
// This file is part of DOLFINX (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include <Eigen/Dense>
#include <array>
#include <cstdint>
#include <dolfinx/graph/AdjacencyList.h>
#include <dolfinx/graph/Partitioning.h>
#include <memory>
#include <vector>

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

  /// Get the permutation numbers to apply to facets.
  /// The permutations are numbered so that:
  ///   n%2 gives the number of reflections to apply
  ///   n//2 gives the number of rotations to apply
  /// Each column of the returned array represents a cell, and each row a
  /// facet of that cell.
  /// This data is used to permute the quadrature point on facet integrals when
  /// data from the cells on both sides of the facet is used.
  /// @return An array of permutation numbers.
  const Eigen::Array<std::uint8_t, Eigen::Dynamic, Eigen::Dynamic>&
  get_facet_permutations() const;

  /// Get the permutation information about the entities of each cell, relative
  /// to a low-to-high ordering. This data is packed so that a 32 bit int is
  /// used for each cell. For 2D cells, one bit is used for each edge, to
  /// represent whether or not the edge is reversed: the least significant bit
  /// is for edge 0, the next for edge 1, etc. For 3D cells, three bits are used
  /// for each face, and for each edge: the least significant bit says whether
  /// or not face 0 is reflected, the next 2 bits say how many times face 0 is
  /// rotated; the next three bits are for face 1, then three for face 2, etc;
  /// after all the faces, there is 1 bit for each edge to say whether or not
  /// they are reversed.
  ///
  /// For example, if a quadrilateral has cell permutation info
  /// ....0111 then (from right to left):
  ///
  ///   - edge 0 is reflected (1)
  ///   - edge 1 is reflected (1)
  ///   - edge 2 is reflected (1)
  ///   - edge 3 is not permuted (0)
  ///
  /// and if a tetrahedron has cell permutation info
  /// ....011010010101001000 then (from right to left):
  ///
  ///   - face 0 is not permuted (000)
  ///   - face 1 is reflected (001)
  ///   - face 2 is rotated twice then reflected (101)
  ///   - face 3 is rotated once (010)
  ///   - edge 0 is not permuted (0)
  ///   - edge 1 is reflected (1)
  ///   - edge 2 is not permuted (0)
  ///   - edge 3 is reflected (1)
  ///   - edge 4 is reflected (1)
  ///   - edge 5 is not permuted (0)
  ///
  /// This data is used to correct the direction of vector function on permuted
  /// facets.
  /// @return A vector of cell permutation info ints.
  const Eigen::Array<std::uint32_t, Eigen::Dynamic, 1>&
  get_cell_permutation_info() const;

  /// Compute entity permutations and reflections used in assembly
  void create_entity_permutations(Topology& topology);

private:
  // The facet permutations
  Eigen::Array<std::uint8_t, Eigen::Dynamic, Eigen::Dynamic>
      _facet_permutations;

  // Cell permutation info
  // See the documentation for get_cell_permutation_info for documentation of
  // how this is encoded.
  Eigen::Array<std::uint32_t, Eigen::Dynamic, 1> _cell_permutation_info;
};

} // namespace mesh
} // namespace dolfinx
