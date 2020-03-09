// Copyright (C) 2006-2019 Anders Logg and Garth N. Wells
//
// This file is part of DOLFINX (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include "cell_types.h"
#include <Eigen/Dense>
#include <array>
#include <cstdint>
#include <dolfinx/common/MPI.h>
#include <memory>
#include <vector>

namespace dolfinx
{
namespace common
{
class IndexMap;
}

namespace fem
{
class ElementDofLayout;
}

namespace graph
{
template <typename T>
class AdjacencyList;
}

namespace mesh
{

class Topology;

/// Compute marker for owned facets that are interior, i.e. are
/// connected to two cells, one of which might be on a remote process
/// @param[in] topology The topology
/// @return Vector with length equal to the number of facets on this
///   this process. True if the ith facet (local index) is interior to
///   the domain.
std::vector<bool> compute_interior_facets(const Topology& topology);

/// Topology stores the topology of a mesh, consisting of mesh entities
/// and connectivity (incidence relations for the mesh entities). Note
/// that the mesh entities don't need to be stored, only the number of
/// entities and the connectivity. Any numbering scheme for the mesh
/// entities is stored separately in a MeshFunction over the entities.
///
/// A mesh entity e may be identified globally as a pair e = (dim, i),
/// where dim is the topological dimension and i is the index of the
/// entity within that topological dimension.

class Topology
{
public:
  /// Create empty mesh topology
  Topology(mesh::CellType type);

  /// Copy constructor
  Topology(const Topology& topology) = default;

  /// Move constructor
  Topology(Topology&& topology) = default;

  /// Destructor
  ~Topology() = default;

  /// Assignment
  Topology& operator=(const Topology& topology) = default;

  /// Assignment
  Topology& operator=(Topology&& topology) = default;

  /// Return topological dimension
  int dim() const;

  /// @todo Merge withset_connectivity
  /// Set the IndexMap for dimension dim
  /// @warning This is experimental and likely to change
  void set_index_map(int dim,
                     std::shared_ptr<const common::IndexMap> index_map);

  /// Get the IndexMap that described the parallel distrubtion of the
  /// mesh entities
  /// @param[in] dim Topological dimension
  /// @return Index map for the entities of dimension @p dim
  std::shared_ptr<const common::IndexMap> index_map(int dim) const;

  /// Marker for entities of dimension dim on the boundary. An entity of
  /// co-dimension < 0 is on the boundary if it is connected to a
  /// boundary facet. It is not defined for codimension 0.
  /// @param[in] dim Toplogical dimension of the entities to check. It
  ///   must be less than the topological dimension.
  /// @return Vector of length equal to number of local entities, with
  ///   'true' for entities on the boundary and otherwise 'false'.
  std::vector<bool> on_boundary(int dim) const;

  /// Return connectivity from entities of dimension d0 to entities of
  /// dimension d1
  /// @param[in] d0
  /// @param[in] d1
  /// @return The adjacency list that for each entity of dimension d0
  ///   gives the list of incident entities of dimension d1
  std::shared_ptr<const graph::AdjacencyList<std::int32_t>>
  connectivity(int d0, int d1) const;

  /// Return connectivity from entities of dimension d0 to entities of
  /// dimension d1
  /// @param[in] d0
  /// @param[in] d1
  /// @return The adjacency list that for each entity of dimension d0
  ///   gives the list of incident entities of dimension d1
  std::shared_ptr<graph::AdjacencyList<std::int32_t>> connectivity(int d0,
                                                                   int d1);

  /// @todo Merge with set_index_map
  /// Set connectivity for given pair of topological dimensions
  void set_connectivity(std::shared_ptr<graph::AdjacencyList<std::int32_t>> c,
                        int d0, int d1);

  /// Gets markers for owned facets that are interior, i.e. are
  /// connected to two cells, one of which might be on a remote process
  /// @return Vector with length equal to the number of facets owned by
  ///   this process. True if the ith facet (local index) is interior to
  ///   the domain.
  const std::vector<bool>& interior_facets() const;

  /// Set markers for owned facets that are interior
  /// @param[in] interior_facets The marker vector
  void set_interior_facets(const std::vector<bool>& interior_facets);

  /// Return hash based on the hash of cell-vertex connectivity
  size_t hash() const;

  /// Cell type
  /// @return Cell type that th topology is for
  mesh::CellType cell_type() const;

  /// Return informal string representation (pretty-print)
  std::string str(bool verbose) const;

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

  /// Resize the arrays of permutations and reflections
  /// @param[in] cell_count The number of cells in the mesh
  /// @param[in] edges_per_cell The number of edges per mesh cell
  /// @param[in] faces_per_cell The number of faces per mesh cell
  void resize_entity_permutations(std::int32_t cell_count, int edges_per_cell,
                                  int faces_per_cell);

  /// Retuns the number of rows in the entity_permutations array
  std::int32_t entity_reflection_size() const;

  /// Set the entity permutations array
  /// @param[in] cell The cell index
  /// @param[in] entity_dim The topological dimension of the entity
  /// @param[in] entity_index The entity number
  /// @param[in] rots The number of rotations to be applied
  /// @param[in] refs The number of reflections to be applied
  void set_entity_permutation(std::int32_t cell, int entity_dim,
                              int entity_index, std::uint8_t rots,
                              std::uint8_t refs);

private:
  // Cell type
  mesh::CellType _cell_type;

  // IndexMap to store ghosting for each entity dimension
  std::array<std::shared_ptr<const common::IndexMap>, 4> _index_map;

  // AdjacencyList for pairs of topological dimensions
  Eigen::Array<std::shared_ptr<graph::AdjacencyList<std::int32_t>>,
               Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
      _connectivity;

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

  // Marker for owned facets, which evaluates to True for facets that
  // are interior to the domain
  std::shared_ptr<const std::vector<bool>> _interior_facets;
};

/// @todo Avoid passing ElementDofLayout. All we need is way to extract
/// the vertices from cells, and the CellType
///
/// Create distributed topology
/// @param[in] comm MPI communicator across which the topology is
///   distributed
/// @param[in] cells The cell topology (list of cell 'nodes') in DOLFIN
///   ordering and using global indices for the nodes. It contains cells
///   that extist only on this this rank and which which have not yet
///   been distributed via a graph partitioner. The input is typically
///   direct from a mesh generator or from file. Cells will be
///   distributed to other ranks.
/// @param[in] layout Describe the association between 'nodes' in @p
///   cells and geometry degrees-of-freedom on the element. It is used
///   to extract the vertex entries in @p cells.
/// @return A distributed Topology, the source rank for each cell in the
///   new topology, and the destination ranks for each cell in @p cells.
std::tuple<Topology, std::vector<int>, graph::AdjacencyList<std::int32_t>>
create_topology(MPI_Comm comm, const graph::AdjacencyList<std::int64_t>& cells,
                const fem::ElementDofLayout& layout);
} // namespace mesh
} // namespace dolfinx
