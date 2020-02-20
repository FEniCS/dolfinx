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
#include <dolfinx/graph/AdjacencyList.h>
#include <memory>
#include <vector>

namespace dolfinx
{
namespace common
{
class IndexMap;
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

  /// Return topological dimension
  int dim() const;

  /// @todo Remove this function. Use IndexMap instead
  /// Set the global indices for entities of dimension dim
  void
  set_global_user_vertices(const std::vector<std::int64_t>& vertex_indices);

  /// Set the IndexMap for dimension dim
  /// @warning This is experimental and likely to change
  void set_index_map(int dim,
                     std::shared_ptr<const common::IndexMap> index_map);

  /// Get the IndexMap for dimension dim
  /// (Currently partially working)
  std::shared_ptr<const common::IndexMap> index_map(int dim) const;

  /// @todo Remove/revise this function. Use IndexMap instead.
  /// Get local-to-global index map for entities of topological
  /// dimension d
  const std::vector<std::int64_t>& get_global_user_vertices() const;

  /// Marker for entities of dimension dim on the boundary. An entity of
  /// co-dimension < 0 is on the boundary if it is connected to a
  /// boundary facet. It is not defined for codimension 0.
  /// @param[in] dim Toplogical dimension of the entities to check. It
  ///   must be less than the topological dimension.
  /// @return Vector of length equal to number of local entities, with
  ///   'true' for entities on the boundary and otherwise 'false'.
  std::vector<bool> on_boundary(int dim) const;

  /// Return connectivity for given pair of topological dimensions
  std::shared_ptr<graph::AdjacencyList<std::int32_t>> connectivity(int d0,
                                                                   int d1);

  /// Return connectivity for given pair of topological dimensions
  std::shared_ptr<const graph::AdjacencyList<std::int32_t>>
  connectivity(int d0, int d1) const;

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

  // TODO: Use std::vector<int32_t> to store 1/0 marker for each edge/face
  /// Get an array of bools that say whether each edge needs to be
  /// reflected to match the low->high ordering of the cell.
  /// @param[in] cell_n The index of the cell.
  /// @return An Eigen::Array of bools
  Eigen::Ref<const Eigen::Array<bool, 1, Eigen::Dynamic>>
  get_edge_reflections(const std::int32_t cell_n) const;

  // TODO: Use std::vector<int32_t> to store 1/0 marker for each edge/face
  /// Get an array of bools that say whether each face needs to be
  /// reflected to match the low->high ordering of the cell.
  /// @param[in] cell_n The index of the cell.
  /// @return An Eigen::Array of bools
  Eigen::Ref<const Eigen::Array<bool, 1, Eigen::Dynamic>>
  get_face_reflections(const std::int32_t cell_n) const;

  /// Get an array of numbers that say how many times each face needs to be
  /// rotated to match the low->high ordering of the cell.
  /// @param[in] cell_n The index of the cell.
  /// @return An Eigen::Array of uint8_ts
  Eigen::Ref<const Eigen::Array<std::uint8_t, 1, Eigen::Dynamic>>
  get_face_rotations(const std::int32_t cell_n) const;

  /// Get the permutation number to apply to a facet.
  /// The permutations are numbered so that:
  ///   n%2 gives the number of reflections to apply
  ///   n//2 gives the number of rotations to apply
  /// @param[in] cell_n The index of the cell
  /// @param[in] dim The dimension of the facet
  /// @param[in] facet_index The local index of the facet
  /// @return The permutation number
  std::uint8_t get_facet_permutation(const std::int32_t cell_n, const int dim,
                                     const int facet_index) const;

  /// Resize the arrays of permutations and reflections
  /// @param[in] cell_count The number of cells in the mesh
  /// @param[in] edges_per_cell The number of edges per mesh cell
  /// @param[in] faces_per_cell The number of faces per mesh cell
  void resize_entity_permutations(std::size_t cell_count, int edges_per_cell,
                                  int faces_per_cell);

  /// Retuns the number of rows in the entity_permutations array
  std::size_t entity_reflection_size() const;

  /// Set the entity permutations array
  /// @param[in] cell_n The cell index
  /// @param[in] entity_dim The topological dimension of the entity
  /// @param[in] entity_index The entity number
  /// @param[in] rots The number of rotations to be applied
  /// @param[in] refs The number of reflections to be applied
  void set_entity_permutation(std::size_t cell_n, int entity_dim,
                              std::size_t entity_index, std::uint8_t rots,
                              std::uint8_t refs);

  /// @todo Move this outside of this class
  /// Set global number of connections for each local entities
  void set_global_size(std::array<int, 2> d,
                       const Eigen::Array<std::int32_t, Eigen::Dynamic, 1>&
                           num_global_connections)
  {
    // assert(num_global_connections.size() == _offsets.size() - 1);
    _num_global_connections(d[0], d[1]) = num_global_connections;
  }

  /// @todo Can this be removed?
  /// Return global number of connections for given entity
  int size_global(std::array<int, 2> d, std::int32_t entity) const
  {
    if (_num_global_connections(d[0], d[1]).size() == 0)
      return _connectivity(d[0], d[1])->num_links(entity);
    else
      return _num_global_connections(d[0], d[1])[entity];
  }

private:
  // Cell type
  mesh::CellType _cell_type;

  // Global indices for vertices
  std::vector<std::int64_t> _global_user_vertices;

  // IndexMap to store ghosting for each entity dimension
  std::array<std::shared_ptr<const common::IndexMap>, 4> _index_map;

  // AdjacencyList for pairs of topological dimensions
  Eigen::Array<std::shared_ptr<graph::AdjacencyList<std::int32_t>>,
               Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
      _connectivity;

  // TODO: Use std::vector<int32_t> to store 1/0 marker for each edge/face
  // The entity reflections of edges
  Eigen::Array<bool, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
      _edge_reflections;

  // TODO: Use std::vector<int32_t> to store 1/0 marker for each edge/face
  // The entity reflections of faces
  Eigen::Array<bool, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
      _face_reflections;

  // The entity reflections of faces
  Eigen::Array<std::uint8_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
      _face_rotations;

  // The entity permutations
  Eigen::Array<std::uint8_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
      _face_permutations;

  // TODO: revise
  // Global number of connections for each entity (possibly not
  // computed)
  Eigen::Array<Eigen::Array<std::int32_t, Eigen::Dynamic, 1>, 4, 4>
      _num_global_connections;
  // Marker for owned facets, which evaluates to True for facets that
  // are interior to the domain
  std::shared_ptr<const std::vector<bool>> _interior_facets;
};
} // namespace mesh
} // namespace dolfinx
