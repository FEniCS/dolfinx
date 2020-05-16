// Copyright (C) 2020 Michal Habera
//
// This file is part of DOLFINX (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include "Geometry.h"
#include "Mesh.h"
#include "Topology.h"
#include <algorithm>
#include <dolfinx/common/IndexMap.h>
#include <dolfinx/common/UniqueIdGenerator.h>
#include <dolfinx/common/utils.h>
#include <dolfinx/graph/AdjacencyList.h>
#include <dolfinx/graph/Partitioning.h>
#include <dolfinx/io/cells.h>
#include <map>
#include <memory>
#include <utility>
#include <vector>

namespace dolfinx
{
namespace mesh
{

/// A MeshTags are used to associate mesh entities with values. The
/// entity index (local to process) identifies the entity. MeshTags is a
/// sparse data storage class; it allows tags to be associated with an
/// arbitrary subset of mesh entities. An entity can have only one
/// associated tag.
/// @tparam Type
template <typename T>
class MeshTags
{
public:
  /// Create from entities of given dimension on a mesh
  /// @param[in] mesh The mesh on which the tags are associated
  /// @param[in] dim Topological dimension of mesh entities to tag
  /// @param[in] indices std::vector<std::int32> of sorted and unique
  ///   entity indices (indices local to the process)
  /// @param[in] values std::vector<T> of values for each index in
  ///   indices. The size must be equal to the size of @p indices.
  template <typename U, typename V>
  MeshTags(const std::shared_ptr<const Mesh>& mesh, int dim, U&& indices,
           V&& values)
      : _mesh(mesh), _dim(dim), _indices(std::forward<U>(indices)),
        _values(std::forward<V>(values))
  {
    if (indices.size() != values.size())
    {
      throw std::runtime_error(
          "Indices and values arrays must have same size.");
    }
#ifdef DEBUG
    if (!std::is_sorted(_indices.begin(), _indices.end()))
      throw std::runtime_error("MeshTag data is not sorted");
    if (std::adjacent_find(_indices.begin(), _indices.end()) != _indices.end())
      throw std::runtime_error("MeshTag data has duplicates");
#endif
  }

  /// Copy constructor
  MeshTags(const MeshTags& tags) = default;

  /// Move constructor
  MeshTags(MeshTags&& tags) = default;

  /// Destructor
  ~MeshTags() = default;

  /// Move assignment
  MeshTags& operator=(const MeshTags& tags) = default;

  /// Move assignment
  MeshTags& operator=(MeshTags&& tags) = default;

  /// Find all entities with a given tag value
  /// @param[in] value The value
  /// @return Indices of tagged entities
  const Eigen::Array<std::int32_t, Eigen::Dynamic, 1> find(const T value) const
  {
    int n = std::count(_values.begin(), _values.end(), value);
    Eigen::Array<std::int32_t, Eigen::Dynamic, 1> indices(n);
    int counter = 0;
    for (std::int32_t i = 0; i < _values.size(); ++i)
    {
      if (_values[i] == value)
        indices[counter++] = _indices[i];
    }
    return indices;
  }

  /// Indices of tagged mesh entities (local-to-process). The indices
  /// are sorted.
  const std::vector<std::int32_t>& indices() const { return _indices; }

  /// Values attached to mesh entities
  const std::vector<T>& values() const { return _values; }

  /// Return topological dimension of tagged entities
  int dim() const { return _dim; }

  /// Return mesh
  std::shared_ptr<const Mesh> mesh() const { return _mesh; }

  /// Name
  std::string name = "mesh_tags";

  /// Unique ID of the object
  std::size_t id() const { return _unique_id; }

private:
  // Unique identifier
  std::size_t _unique_id = common::UniqueIdGenerator::id();

  // Associated mesh
  std::shared_ptr<const Mesh> _mesh;

  // Topological dimension of tagged mesh entities
  int _dim;

  // Local-to-process indices of tagged entities
  std::vector<std::int32_t> _indices;

  // Values attached to entities
  std::vector<T> _values;
};

/// @todo Generalise to create multiple MeshTags as some of the data sent
/// (expensively) via MPI re-used.
///
/// Create MeshTags from arrays
/// @param[in] comm The MPI communicator
/// @param[in] mesh The Mesh that the tags are associated with
/// @param[in] tag_cell_type Cell type of entities which are being
///   tagged
/// @param[in] entities 'Node' indices (using the of entities
///   input_global_indices from the Mesh) for vertices of for each
///   entity that is tagged. The numbers of rows is equal to the number
///   of entities.
/// @param[in] values Tag values for each entity in @ entities. The
///   length of @ values  must be equal to number of rows in @ entities.
template <typename T>
mesh::MeshTags<T>
create_meshtags(const std::shared_ptr<const mesh::Mesh>& mesh,
                const mesh::CellType& tag_cell_type,
                const graph::AdjacencyList<std::int64_t>& entities,
                const std::vector<T>& values)
{
  // TODO: Avoid expensive-to-create std::vector<std::vector>>. Build
  //       AdjacencyList instead.

  assert(mesh);
  if ((std::size_t)entities.num_nodes() != values.size())
    throw std::runtime_error("Number of entities and values must match");

  // Tagged entity topological dimension
  const int e_dim = mesh::cell_dim(tag_cell_type);

  const auto map_v = mesh->topology().index_map(0);
  assert(map_v);
  const auto map_e = mesh->topology().index_map(e_dim);
  assert(map_e);

  auto e_to_v = mesh->topology().connectivity(e_dim, 0);
  if (!e_to_v)
    throw std::runtime_error("Missing entity-vertex connectivity.");

  // -------------------
  // 1. For each entity on this rank, send its lowest global vertex index to the
  //    'postmaster' rank, and receive global vertex indices for which
  //    this rank is the postmaster

  const int comm_size = MPI::size(mesh->mpi_comm());
  const int num_vertices_per_entity = e_to_v->num_links(0);
  const int num_entities_mesh = map_e->size_local() + map_e->num_ghosts();
  std::vector<std::vector<std::int64_t>> entity_vertex0_send(comm_size);

  std::map<std::vector<std::int64_t>, std::int32_t> entity_key_to_index;
  std::vector<std::int64_t> key(num_vertices_per_entity);
  const std::vector<std::int64_t> global_indices = map_v->global_indices();
  const std::size_t num_vertices_global = map_v->size_global();

  for (int e = 0; e < num_entities_mesh; ++e)
  {
    // Prepare a map from ordered global vertex indices to local entity number
    // This map is used later to identify if received entity is owned or ghost
    auto vertices = e_to_v->links(e);
    for (int v = 0; v < vertices.rows(); ++v)
      key[v] = global_indices[vertices[v]];
    std::sort(key.begin(), key.end());
    entity_key_to_index.insert({key, e});

    const int p
        = dolfinx::MPI::index_owner(comm_size, key[0], num_vertices_global);
    entity_vertex0_send[p].push_back(key[0]);
  }

  const graph::AdjacencyList<std::int64_t> entity_vertex0_recv = MPI::all_to_all(
      mesh->mpi_comm(), graph::AdjacencyList<std::int64_t>(entity_vertex0_send));

  // -------------------
  // 2. Send the entity key (vertex list) and tag to the postmaster based
  //    on the lowest index vertex in the entity 'key'
  //
  //    NOTE: Stage 2 doesn't depend on the data received in Step 1, so
  //    data (i) the communication could be combined, or (ii) the
  //    communication in Step 1 could be make non-blocking.

  std::vector<std::vector<std::int64_t>> entities_send(comm_size);
  std::vector<std::vector<T>> values_send(comm_size);
  std::vector<std::int64_t> entity(num_vertices_per_entity);
  for (std::int32_t e = 0; e < entities.num_nodes(); ++e)
  {
    // Copy vertices for entity and sort
    std::copy(entities.links(e).data(),
              entities.links(e).data() + entities.num_links(e), entity.begin());
    std::sort(entity.begin(), entity.end());

    const int p = dolfinx::MPI::index_owner(comm_size, entity[0], num_vertices_global);
    entities_send[p].insert(entities_send[p].end(), entity.begin(),
                                   entity.end());
    values_send[p].push_back(values[e]);
  }

  // TODO: Pack into one MPI call
  const graph::AdjacencyList<std::int64_t> entities_recv = MPI::all_to_all(
      mesh->mpi_comm(), graph::AdjacencyList<std::int64_t>(entities_send));
  const graph::AdjacencyList<T> values_recv
      = MPI::all_to_all(mesh->mpi_comm(), graph::AdjacencyList<T>(values_send));

  // -------------------
  // 3. As 'postmaster', send back the entity key (vertex list) and tag
  //    value to ranks that possibly need the data. Do this based on the
  //    first vertex index in the entity key.

  // NOTE: Could: (i) use a std::unordered_multimap, or (ii) only send
  // owned vertices to the postmaster and use map, unordered_map or
  // std::vector<pair>>, followed by a neighbourhood all_to_all at the
  // end.
  //
  // Build map from global vertex index to ranks that need the facet which has
  // this vertex
  std::multimap<std::int64_t, int> vertex_to_rank;
  for (int p = 0; p < entity_vertex0_recv.num_nodes(); ++p)
  {
    auto vertices = entity_vertex0_recv.links(p);
    for (int v = 0; v < vertices.rows(); ++v)
      vertex_to_rank.insert({vertices(v), p});
  }

  // Figure out which processes are owners of received vertices
  std::vector<std::vector<std::int64_t>> send_vertices_owned(comm_size);
  std::vector<std::vector<T>> send_vals_owned(comm_size);
  const Eigen::Map<const Eigen::Array<std::int64_t, Eigen::Dynamic,
                                      Eigen::Dynamic, Eigen::RowMajor>>
      _entities_recv(entities_recv.array().data(),
                     entities_recv.array().rows() / num_vertices_per_entity,
                     num_vertices_per_entity);
  auto _values_recv = values_recv.array();
  assert(_values_recv.rows() == _entities_recv.rows());
  for (int e = 0; e < _entities_recv.rows(); ++e)
  {
    // Find ranks that have vertex0
    auto [it0, it1] = vertex_to_rank.equal_range(_entities_recv(e, 0));
    for (auto it = it0; it != it1; ++it)
    {
      const int p1 = it->second;
      send_vertices_owned[p1].insert(
          send_vertices_owned[p1].end(), _entities_recv.row(e).data(),
          _entities_recv.row(e).data() + _entities_recv.cols());
      send_vals_owned[p1].push_back(_values_recv(e));
    }
  }

  // TODO: Pack into one MPI call
  const graph::AdjacencyList<std::int64_t> recv_ents = MPI::all_to_all(
      mesh->mpi_comm(), graph::AdjacencyList<std::int64_t>(send_vertices_owned));
  const graph::AdjacencyList<T> recv_vals
      = MPI::all_to_all(mesh->mpi_comm(), graph::AdjacencyList<T>(send_vals_owned));

  // Iterate over all received entities. If entity is on this rank,
  // store (local entity index, tag value)
  std::vector<std::int32_t> indices_new;
  std::vector<T> values_new;
  const Eigen::Map<const Eigen::Array<std::int64_t, Eigen::Dynamic,
                                      Eigen::Dynamic, Eigen::RowMajor>>
      _entities(recv_ents.array().data(),
                recv_ents.array().rows() / num_vertices_per_entity,
                num_vertices_per_entity);
  entity.resize(num_vertices_per_entity);

  for (Eigen::Index e = 0; e < _entities.rows(); ++e)
  {
    // Note: _entities.row(e) was sorted by the sender
    std::copy(_entities.row(e).data(),
              _entities.row(e).data() + num_vertices_per_entity, entity.begin());
    if (const auto it = entity_key_to_index.find(entity);
        it != entity_key_to_index.end())
    {
      indices_new.push_back(it->second);
      values_new.push_back(recv_vals.array()[e]);
    }
  }

  auto [indices_sorted, values_sorted]
      = common::sort_unique(indices_new, values_new);
  return mesh::MeshTags<T>(mesh, e_dim, std::move(indices_sorted),
                       std::move(values_sorted));
}
} // namespace mesh
} // namespace dolfinx
