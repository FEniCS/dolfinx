// Copyright (C) 2020 Michal Habera
//
// This file is part of DOLFINX (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include "Geometry.h"
#include "Mesh.h"
#include "Topology.h"
#include <dolfinx/common/IndexMap.h>
#include <dolfinx/common/UniqueIdGenerator.h>
#include <dolfinx/graph/AdjacencyList.h>
#include <dolfinx/graph/Partitioning.h>
#include <dolfinx/io/cells.h>
#include <map>
#include <memory>
#include <unordered_map>
#include <utility>
#include <vector>

namespace dolfinx
{
namespace mesh
{

/// A MeshTags is a class used to associate mesh entities with values.
/// The entity index (local to process) identifies the entity. MeshTags
/// is a sparse data storage class; it allows tags to be associated with
/// an arbitrary subset of mesh entities. An entity can have only one
/// associated tag.
/// @tparam Type
template <typename T>
class MeshTags
{
public:
  /// Create from entities of given dimension on a mesh
  /// @param[in] mesh The mesh associated with the tags
  /// @param[in] dim Topological dimension of mesh entities to tag
  /// @param[in] indices std::vector<std::int32> of entity indices
  ///   (indices local to the process)
  /// @param[in] values std::vector<T> of values for each index in
  ///   indices
  /// @param[in] sorted True for already sorted indices
  /// @param[in] unique True for unique indices
  template <typename U, typename V>
  MeshTags(const std::shared_ptr<const Mesh>& mesh, int dim, U&& indices,
           V&& values, const bool sorted = false, const bool unique = false)
      : _mesh(mesh), _dim(dim), _indices(std::forward<U>(indices)),
        _values(std::forward<V>(values))
  {
    if (indices.size() != values.size())
    {
      throw std::runtime_error(
          "Indices and values arrays must have same size.");
    }
    if (!sorted)
      sort();
    if (!unique)
      remove_duplicates();
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

  /// Indices of tagged mesh entities, local-to-process (const version)
  const std::vector<std::int32_t>& indices() const { return _indices; }

  /// Values attached to mesh entities (const.)
  const std::vector<T>& values() const { return _values; }

  /// Return topological dimension of tagged entities
  int dim() const { return _dim; }

  /// Return mesh
  std::shared_ptr<const Mesh> mesh() const { return _mesh; }

  /// Name
  std::string name = "mesh_tags";

  /// Unique ID
  std::size_t id() const { return _unique_id; }

private:
  // Unique identifier
  std::size_t _unique_id = common::UniqueIdGenerator::id();

  /// Associated mesh
  std::shared_ptr<const Mesh> _mesh;

  /// Topological dimension of tagged mesh entities
  int _dim;

  // Local-to-process indices of tagged entities
  std::vector<std::int32_t> _indices;

  // Values attached to entities
  std::vector<T> _values;

  // Sort indices and values according by index
  void sort()
  {
    // Compute the sorting permutation
    std::vector<int> perm(_indices.size());
    std::iota(perm.begin(), perm.end(), 0);
    std::sort(perm.begin(), perm.end(),
              [&indices = std::as_const(_indices)](const int a, const int b) {
                return (indices[a] < indices[b]);
              });

    // Copy data
    const std::vector<std::int32_t> indices_tmp = _indices;
    const std::vector<T> values_tmp = _values;

    // Apply sorting and insert
    for (std::size_t i = 0; i < indices_tmp.size(); ++i)
    {
      _indices[i] = indices_tmp[perm[i]];
      _values[i] = values_tmp[perm[i]];
    }
  }

  // Remove duplicates in indices and values according to indices
  void remove_duplicates()
  {
    // Algorithm would fail for empty vector
    if (_indices.size() == 0)
      return;

    std::size_t last_unique = 0;
    for (std::size_t i = 0; i < _indices.size(); ++i)
    {
      if (_indices[i] > _indices[last_unique])
      {
        _indices[++last_unique] = _indices[i];
        _values[last_unique] = _values[i];
      }
    }
    _indices.erase(_indices.begin() + last_unique + 1, _indices.end());
    _values.erase(_values.begin() + last_unique + 1, _values.end());
  }
};

/// Create MeshTags from arrays
/// @param[in] comm The MPI communicator
/// @param[in] mesh The Mesh that the tags are associated with
/// @param[in] tag_cell_type Cell type of entities which are being
///   tagged
/// @param[in] entities 'Node' indices (using the of entities
///   input_global_indices from the Mesh) for each entity this is tagged
///   mesh entities. The numbers of rows is equal to the number of
///   entities.
/// @param[in] values Tag values for each entity in @ entities. The
/// length of @ values  must be equal to number of rows in @ entities.
template <typename T>
mesh::MeshTags<T>
create_meshtags(MPI_Comm comm, const std::shared_ptr<const mesh::Mesh>& mesh,
                const mesh::CellType& tag_cell_type,
                const Eigen::Array<std::int64_t, Eigen::Dynamic, Eigen::Dynamic,
                                   Eigen::RowMajor>& entities,
                const std::vector<T>& values)
{
  // TODO: Avoid expensive-to-create std::vector<std::vector>>. Build
  // AdjacencyList instead.

  assert(mesh);
  if ((std::size_t)entities.rows() != values.size())
    throw std::runtime_error("Number of entities and values must match");

  // Build sorted array of entity node indices
  std::vector<std::int64_t> nodes(
      entities.data(), entities.data() + entities.rows() * entities.cols());
  std::sort(nodes.begin(), nodes.end());
  nodes.erase(std::unique(nodes.begin(), nodes.end()), nodes.end());

  // Get mesh connectivity
  const int e_dim = mesh::cell_dim(tag_cell_type);
  const int dim = mesh->topology().dim();
  auto e_to_v = mesh->topology().connectivity(e_dim, 0);
  if (!e_to_v)
    throw std::runtime_error("Missing entity-vertex connectivity.");
  auto e_to_c = mesh->topology().connectivity(e_dim, dim);
  if (!e_to_c)
    throw std::runtime_error("Missing entity-cell connectivity.");
  auto c_to_v = mesh->topology().connectivity(dim, 0);
  if (!c_to_v)
    throw std::runtime_error("missing cell-vertex connectivity.");

  // -------------------
  // 1. Send this rank's global "input" nodes indices to the
  //    'postmaster' rank, and receive global "input" nodes for which
  //    this rank is the postmaster

  // Get "input" global node indices (as in the input file before any
  // internal re-ordering)
  const std::vector<std::int64_t>& nodes_g
      = mesh->geometry().input_global_indices();

  // Send input global indices to 'post master' rank, based on input
  // global index value
  const std::int64_t num_nodes_g = mesh->geometry().index_map()->size_global();
  const int comm_size = MPI::size(comm);
  // NOTE: could make this int32_t be sending: index <- index - dest_rank_offset
  std::vector<std::vector<std::int64_t>> nodes_g_send(comm_size);
  for (std::int64_t node : nodes_g)
  {
    // TODO: Optimise this call by adding 'vectorised verion of
    //       MPI::index_owner
    // Figure out which process is the postmaster for the input global index
    const std::int32_t p
        = dolfinx::MPI::index_owner(comm_size, node, num_nodes_g);
    nodes_g_send[p].push_back(node);
  }

  // Send/receive
  const graph::AdjacencyList<std::int64_t> nodes_g_recv
      = MPI::all_to_all(comm, graph::AdjacencyList<std::int64_t>(nodes_g_send));

  // -------------------
  // 2. Send the entity key (nodes list) and tag to the postmaster based
  //    on the lowest index node in the entity 'key'
  //
  //    NOTE: Stage 2 doesn't depend on the data received in Step 1, so
  //    data (i) the communication could be combined, or (ii) the
  //    communication in Step 1 could be make non-blocking.

  std::vector<std::vector<std::int64_t>> entities_send(comm_size);
  std::vector<std::vector<T>> values_send(comm_size);
  for (std::int32_t e = 0; e < entities.rows(); ++e)
  {
    // Copy nodes for entity
    std::vector<std::int64_t> entity(entities.row(e).data(),
                                     entities.row(e).data() + entities.cols());
    std::sort(entity.begin(), entity.end());

    // Determine postmaster based on lowest entity node
    const std::int32_t p
        = dolfinx::MPI::index_owner(comm_size, entity[0], num_nodes_g);
    entities_send[p].insert(entities_send[p].end(), entity.begin(),
                            entity.end());
    values_send[p].push_back(values[e]);
  }

  // TODO: Pack into one MPI call
  const graph::AdjacencyList<std::int64_t> entities_recv = MPI::all_to_all(
      comm, graph::AdjacencyList<std::int64_t>(entities_send));
  const graph::AdjacencyList<T> values_recv
      = MPI::all_to_all(comm, graph::AdjacencyList<T>(values_send));

  // -------------------
  // 3. As 'postmaster', send back the entity key (node list) and tag
  //    value to ranks that possibly need the data. Do this based on the
  //    first node index in the entity key.

  // NOTE: Could: (i) use a std::unordered_multimap, or (ii) only send
  // owned nodes to the postmaster and use map, unordered_map or
  // std::vector<pair>>, followed by a neighbourhood all_to_all at the
  // end.

  // Build map from global node index to ranks that have it
  std::multimap<std::int64_t, int> node_to_rank;
  for (int p = 0; p < nodes_g_recv.num_nodes(); ++p)
  {
    auto nodes = nodes_g_recv.links(p);
    for (int i = 0; i < nodes.rows(); ++i)
      node_to_rank.insert({nodes(i), p});
  }

  // Figure out which processes are owners of received nodes
  std::vector<std::vector<std::int64_t>> send_nodes_owned(comm_size);
  std::vector<std::vector<T>> send_vals_owned(comm_size);
  const int nnodes_per_entity = entities.cols();

  // Loop over process ranks
  for (int p = 0; p < entities_recv.num_nodes(); ++p)
  {
    auto nodes = entities_recv.links(p);
    auto vals = values_recv.links(p);

    // Loop over received entities from rank p
    for (int e = 0; e < nodes.size() / nnodes_per_entity; ++e)
    {
      std::vector<std::int64_t> entity(nodes.data() + e * nnodes_per_entity,
                                       nodes.data() + e * nnodes_per_entity
                                           + nnodes_per_entity);

      // Find ranks that have node0
      auto [it0, it1] = node_to_rank.equal_range(entity[0]);
      for (auto it = it0; it != it1; ++it)
      {
        const int p1 = it->second;
        send_nodes_owned[p1].insert(send_nodes_owned[p1].end(), entity.begin(),
                                    entity.end());
        send_vals_owned[p1].push_back(vals(e));
      }

      // // Loop over process ranks
      // for (int p1 = 0; p1 < nodes_g_recv.num_nodes(); ++p1)
      // {
      //   auto igi = nodes_g_recv.links(p1);
      //   for (int j = 0; j < igi.size(); ++j)
      //   {
      //     if (igi[j] == entity[0])
      //     {
      //       send_nodes_owned[p1].insert(send_nodes_owned[p1].end(),
      //                                   entity.begin(), entity.end());
      //       send_vals_owned[p1].push_back(vals(e));
      //     }
      //   }
      // }

      // // Loop over process ranks
      // for (int p1 = 0; p1 < nodes_g_recv.num_nodes(); ++p1)
      // {
      //   auto igi = nodes_g_recv.links(p1);
      //   for (int j = 0; j < igi.size(); ++j)
      //   {
      //     if (igi[j] == entity[0])
      //     {
      //       send_nodes_owned[p1].insert(send_nodes_owned[p1].end(),
      //                                   entity.begin(), entity.end());
      //       send_vals_owned[p1].push_back(vals(e));
      //     }
      //   }
      // }
    }
  }

  // TODO: Pack into one MPI call
  const graph::AdjacencyList<std::int64_t> recv_ents = MPI::all_to_all(
      comm, graph::AdjacencyList<std::int64_t>(send_nodes_owned));
  const graph::AdjacencyList<T> recv_vals
      = MPI::all_to_all(comm, graph::AdjacencyList<T>(send_vals_owned));

  // -------------------
  // 4. From the received (key, value) data, determine which keys
  //    (entities) are on this process.

  // Using just the information on current local mesh partition prepare
  // a mapping from *ordered* nodes of entity input global indices to
  // entity local index
  std::map<std::vector<std::int64_t>, std::int32_t> entities_igi;
  auto map_e = mesh->topology().index_map(e_dim);
  assert(map_e);
  const std::int32_t num_entities = map_e->size_local() + map_e->num_ghosts();
  const graph::AdjacencyList<std::int32_t>& cells_g = mesh->geometry().dofmap();
  const std::vector<std::uint8_t> vtk_perm
      = io::cells::vtk_to_dolfin(tag_cell_type, nnodes_per_entity);
  for (std::int32_t e = 0; e < num_entities; ++e)
  {
    std::vector<std::int64_t> entity_igi(nnodes_per_entity);

    // Find cell attached to the entity
    std::int32_t c = e_to_c->links(e)[0];
    auto cell_nodes = cells_g.links(c);
    auto cell_vertices = c_to_v->links(c);
    auto entity_vertices = e_to_v->links(e);
    for (int v = 0; v < entity_vertices.rows(); ++v)
    {
      // Find local index of vertex wrt cell
      const std::int32_t vertex = entity_vertices[vtk_perm[v]];
      auto it = std::find(cell_vertices.data(),
                          cell_vertices.data() + cell_vertices.rows(), vertex);
      assert(it != (cell_vertices.data() + cell_vertices.rows()));
      const int local_cell_vertex = std::distance(cell_vertices.data(), it);

      // Insert "input" global index for the node of the entity
      entity_igi[v] = nodes_g[cell_nodes[local_cell_vertex]];
    }

    // Sorting is needed to match with entities stored in file
    std::sort(entity_igi.begin(), entity_igi.end());
    entities_igi.insert({entity_igi, e});
  }

  // Iterate over all received entities and find it in entities of the
  // mesh
  std::vector<std::int32_t> indices_new;
  std::vector<T> values_new;
  const Eigen::Map<const Eigen::Array<std::int64_t, Eigen::Dynamic,
                                      Eigen::Dynamic, Eigen::RowMajor>>
      _entities(recv_ents.array().data(),
                recv_ents.array().rows() / nnodes_per_entity,
                nnodes_per_entity);
  std::vector<std::int64_t> entity(nnodes_per_entity);
  for (Eigen::Index e = 0; e < _entities.rows(); ++e)
  {
    std::copy(_entities.row(e).data(),
              _entities.row(e).data() + nnodes_per_entity, entity.begin());
    std::sort(entity.begin(), entity.end());
    if (const auto it = entities_igi.find(entity); it != entities_igi.end())
    {
      indices_new.push_back(it->second);
      values_new.push_back(recv_vals.array()[e]);
    }
  }

  // -------------------
  // 5. Build MeshTags object

  return mesh::MeshTags<T>(mesh, e_dim, std::move(indices_new),
                           std::move(values_new));
}
} // namespace mesh
} // namespace dolfinx
