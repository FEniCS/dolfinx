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
/// @param[in] comm
/// @param[in] mesh
/// @param[in] entity_cell_type Cell type of entities which MeshTags are
///   tagging
/// @param[in] topology Array describing topology of tagged mesh
///   entities. This array must be using input_global_indices from the
///   mesh.
/// @param[in] values Array of values to attach to mesh entities
template <typename T>
mesh::MeshTags<T>
create_meshtags(MPI_Comm comm, const std::shared_ptr<const mesh::Mesh>& mesh,
                const mesh::CellType& entity_cell_type,
                const Eigen::Array<std::int64_t, Eigen::Dynamic, Eigen::Dynamic,
                                   Eigen::RowMajor>& topology,
                const std::vector<T>& values)
{
  if ((std::size_t)topology.rows() != values.size())
    throw std::runtime_error("Number of entities and values must match");

  // Build array of topology node indices
  std::vector<std::int64_t> topo_unique(
      topology.data(), topology.data() + topology.rows() * topology.cols());
  std::sort(topo_unique.begin(), topo_unique.end());
  topo_unique.erase(std::unique(topo_unique.begin(), topo_unique.end()),
                    topo_unique.end());

  const int e_dim = mesh::cell_dim(entity_cell_type);
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

  // Get "input" global node indices (as in the input file before any
  // internal re-ordering)
  const std::vector<std::int64_t>& igi
      = mesh->geometry().input_global_indices();

  // Send input global indices to process responsible for it, based on
  // input global index value
  std::int64_t num_igi_global = 0;
  const std::int64_t igi_size = igi.size();
  MPI_Allreduce(&igi_size, &num_igi_global, 1, MPI_INT64_T, MPI_SUM, comm);

  // Split global array size and retrieve a range that this
  // process/officer is responsible for
  const int comm_size = MPI::size(comm);
  std::array<std::int64_t, 2> range
      = MPI::local_range(MPI::rank(comm), num_igi_global, comm_size);
  const int local_size = range[1] - range[0];
  std::vector<std::vector<std::int64_t>> send_igi(comm_size);
  const std::int32_t size_local = mesh->geometry().index_map()->size_local()
                                  + mesh->geometry().index_map()->num_ghosts();
  assert((std::size_t)size_local == igi.size());
  for (std::int32_t i = 0; i < size_local; ++i)
  {
    // TODO: Optimise this call
    // Figure out which process responsible for the input global index
    const int officer = MPI::index_owner(comm_size, igi[i], num_igi_global);
    send_igi[officer].push_back(igi[i]);
  }

  const graph::AdjacencyList<std::int64_t> recv_igi
      = MPI::all_to_all(comm, graph::AdjacencyList<std::int64_t>(send_igi));

  // Handle received input global indices, i.e. put the owners of it to
  // a global position, which is its value
  Eigen::Array<bool, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> owners(
      local_size, comm_size);
  const std::size_t offset = MPI::global_offset(comm, local_size, true);
  for (std::int32_t i = 0; i < recv_igi.num_nodes(); ++i)
  {
    auto data = recv_igi.links(i);
    for (std::int32_t j = 0; j < data.rows(); ++j)
    {
      const std::int32_t local_index = data[j] - offset;
      assert(local_size > local_index);
      assert(local_index >= 0);
      owners(local_index, i) = true;
    }
  }

  // Distribute the owners of input global indices

  // Distribute owners and fetch owners for the input global indices
  // read from file, i.e. for the unique topology data in file
  const Eigen::Array<bool, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
      dist_owners
      = graph::Partitioning::distribute_data<bool>(comm, topo_unique, owners);

  // Figure out which process needs input global indices read from file
  // and send to it

  // Mapping from global topology number to its ownership (bools saying
  // if the process is owner)
  std::unordered_map<std::int64_t, Eigen::Array<bool, Eigen::Dynamic, 1>>
      topo_owners;
  for (std::size_t i = 0; i < topo_unique.size(); ++i)
    topo_owners[topo_unique[i]] = dist_owners.row(i);

  std::vector<std::vector<std::int64_t>> send_ents(comm_size);
  std::vector<std::vector<T>> send_vals(comm_size);
  const int nnodes_per_entity = topology.cols();
  for (Eigen::Index e = 0; e < topology.rows(); ++e)
  {
    std::vector<std::int64_t> entity(nnodes_per_entity);
    std::vector<bool> sent(comm_size, false);
    for (int i = 0; i < nnodes_per_entity; ++i)
      entity[i] = topology(e, i);

    // Figure out owners of this entity. Entity has several nodes and
    // each node can have up to comm_size owners, need to send the
    // entity to each owner.
    for (int i = 0; i < nnodes_per_entity; ++i)
    {
      for (int j = 0; j < comm_size; ++j)
      {
        if (topo_owners[entity[i]][j] && !sent[j])
        {
          send_ents[j].insert(send_ents[j].end(), entity.begin(), entity.end());
          send_vals[j].push_back(values[e]);
          sent[j] = true;
        }
      }
    }
  }

  // Using just the information on current local mesh partition prepare
  // a mapping from *ordered* nodes of entity input global indices to
  // entity local index
  std::map<std::vector<std::int64_t>, std::int32_t> entities_igi;
  auto map_e = mesh->topology().index_map(e_dim);
  assert(map_e);
  const std::int32_t num_entities = map_e->size_local() + map_e->num_ghosts();
  const graph::AdjacencyList<std::int32_t>& cells_g = mesh->geometry().dofmap();
  const std::vector<std::uint8_t> vtk_perm
      = io::cells::vtk_to_dolfin(entity_cell_type, nnodes_per_entity);
  for (std::int32_t e = 0; e < num_entities; ++e)
  {
    std::vector<std::int64_t> entity_igi(nnodes_per_entity);

    // Iterate over all entities of the mesh. Find cell attached to the
    // entity.
    std::int32_t c = e_to_c->links(e)[0];
    auto cell_nodes = cells_g.links(c);
    auto cell_vertices = c_to_v->links(c);
    auto entity_vertices = e_to_v->links(e);
    for (int v = 0; v < entity_vertices.rows(); ++v)
    {
      // Find local index of vertex wrt. cell
      const std::int32_t vertex = entity_vertices[vtk_perm[v]];
      auto it = std::find(cell_vertices.data(),
                          cell_vertices.data() + cell_vertices.rows(), vertex);
      assert(it != (cell_vertices.data() + cell_vertices.rows()));
      const int local_cell_vertex = std::distance(cell_vertices.data(), it);

      // Insert input global index for the node of the entity
      entity_igi[v] = igi[cell_nodes[local_cell_vertex]];
    }

    // Sorting is needed to match with entities stored in file
    std::sort(entity_igi.begin(), entity_igi.end());
    entities_igi.insert({entity_igi, e});
  }

  const graph::AdjacencyList<std::int64_t> recv_ents
      = MPI::all_to_all(comm, graph::AdjacencyList<std::int64_t>(send_ents));
  const graph::AdjacencyList<T> recv_vals
      = MPI::all_to_all(comm, graph::AdjacencyList<T>(send_vals));

  // Iterate over all received entities and find it in entities of the
  // mesh
  std::vector<std::int32_t> indices_new;
  std::vector<T> values_new;
  const Eigen::Map<const Eigen::Array<std::int64_t, Eigen::Dynamic,
                                      Eigen::Dynamic, Eigen::RowMajor>>
      entities(recv_ents.array().data(),
               recv_ents.array().rows() / nnodes_per_entity, nnodes_per_entity);
  std::vector<std::int64_t> entity(nnodes_per_entity);
  for (Eigen::Index e = 0; e < entities.rows(); ++e)
  {
    std::copy(entities.row(e).data(),
              entities.row(e).data() + nnodes_per_entity, entity.begin());
    std::sort(entity.begin(), entity.end());
    if (const auto it = entities_igi.find(entity); it != entities_igi.end())
    {
      indices_new.push_back(it->second);
      values_new.push_back(recv_vals.array()[e]);
    }
  }

  return mesh::MeshTags<T>(mesh, e_dim, std::move(indices_new),
                           std::move(values_new));
}
} // namespace mesh
} // namespace dolfinx
