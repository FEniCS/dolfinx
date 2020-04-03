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
#include <dolfinx/graph/Partitioning.h>
#include <dolfinx/io/cells.h>
#include <map>
#include <memory>
#include <utility>

namespace dolfinx
{
namespace mesh
{

/// A MeshTags is a class used to tag mesh entities using their
/// local-to-process index and an attached value.
/// MeshTags is a sparse data storage class, since it allows to
/// tag only few mesh entities. This class stores sorted and
/// unique indices.
/// @tparam Type
template <typename T>
class MeshTags
{
public:
  /// Create from entities of given dimension on a mesh
  /// @param[in] mesh The mesh associated with the tags
  /// @param[in] dim Topological dimension of mesh entities
  ///    to tag.
  /// @param[in] indices Array of indices, will be copied, sorted
  ///    with duplicates removed. Local-to-process.
  /// @param[in] values Array of values attached to indices,
  ///    will be copied, sorted and duplicates removed according
  ///    to indices array.
  /// @param[in] sorted True for already sorted indices.
  /// @param[in] unique True for unique indices.
  MeshTags(const std::shared_ptr<const Mesh>& mesh, int dim,
           const std::vector<std::int32_t>& indices,
           const std::vector<T>& values, const bool sorted = false,
           const bool unique = false);

  /// Create from entities of given dimension on a mesh
  /// @param[in] mesh The mesh associated with the tags
  /// @param[in] dim Topological dimension of mesh entities
  ///    to tag.
  /// @param[in] indices Array of indices, will be copied, sorted
  ///    with duplicates removed. Local-to-process.
  /// @param[in] values Array of values attached to indices,
  ///    will be copied, sorted and duplicates removed according
  ///    to indices array.
  /// @param[in] sorted True for already sorted indices.
  /// @param[in] unique True for unique indices.
  MeshTags(const std::shared_ptr<const Mesh>& mesh, int dim,
           std::vector<std::int32_t>&& indices, std::vector<T>&& values,
           const bool sorted = false, const bool unique = false);

  /// Move constructor
  MeshTags(MeshTags&& mt) = default;

  /// Destructor
  ~MeshTags() = default;

  /// Move assignment
  MeshTags& operator=(MeshTags&&) = default;

  /// Indices of tagged mesh entities, local-to-process (const.)
  const std::vector<std::int32_t>& indices() const;

  /// Values attached to mesh entities (const.)
  const std::vector<T>& values() const;

  /// Return topological dimension of tagged entities
  int dim() const;

  /// Return mesh
  std::shared_ptr<const Mesh> mesh() const;

  /// Name
  std::string name = "mesh_tags";

  /// Unique ID
  std::size_t id() const { return _unique_id; }

private:
  // Unique identifier
  std::size_t _unique_id;

  /// Associated mesh
  std::shared_ptr<const Mesh> _mesh;

  /// Topological dimension of tagged mesh entities
  int _dim;

  // Local-to-process indices of tagged entities
  std::vector<std::int32_t> _indices;

  // Values attached to entities
  std::vector<T> _values;

  // Sort indices and values according to indices
  void sort();

  // Remove duplicates in indices and values according to indices
  void remove_duplicates();
};

/// Create MeshTags from arrays
/// @param[in] comm
/// @param[in] mesh
/// @param[in] entity_cell_type Cell type of entities which MeshTags are
///   tagging.
/// @param[in] topology Array describing topology of tagged mesh entities.
///   This array must be using input_global_indices from the mesh.
/// @param[in] values Array of values to attach to mesh entities.
template <typename T>
MeshTags<T>
create_meshtags(MPI_Comm comm, const std::shared_ptr<const mesh::Mesh>& mesh,
                const mesh::CellType& entity_cell_type,
                const Eigen::Array<std::int64_t, Eigen::Dynamic, Eigen::Dynamic,
                                   Eigen::RowMajor>& topology,
                std::vector<T>& values);

//---------------------------------------------------------------------------
// Implementation
//---------------------------------------------------------------------------
template <typename T>
MeshTags<T>::MeshTags(const std::shared_ptr<const Mesh>& mesh, int dim,
                      const std::vector<std::int32_t>& indices,
                      const std::vector<T>& values, const bool sorted,
                      const bool unique)
    : _unique_id(common::UniqueIdGenerator::id()), _mesh(mesh), _dim(dim),
      _indices(indices), _values(values)
{
  if (indices.size() != values.size())
    throw std::runtime_error("Indices and values arrays must match in size.");

  if (!sorted)
    sort();

  if (!unique)
    remove_duplicates();
}
//---------------------------------------------------------------------------
template <typename T>
MeshTags<T>::MeshTags(const std::shared_ptr<const Mesh>& mesh, int dim,
                      std::vector<std::int32_t>&& indices,
                      std::vector<T>&& values, const bool sorted,
                      const bool unique)
    : _unique_id(common::UniqueIdGenerator::id()), _mesh(mesh), _dim(dim),
      _indices(std::move(indices)), _values(std::move(values))
{
  if (indices.size() != values.size())
    throw std::runtime_error("Indices and values arrays must match in size.");

  if (!sorted)
    sort();

  if (!unique)
    remove_duplicates();
}
//---------------------------------------------------------------------------
template <typename T>
const std::vector<T>& MeshTags<T>::values() const
{
  return _values;
}
//---------------------------------------------------------------------------
template <typename T>
const std::vector<std::int32_t>& MeshTags<T>::indices() const
{
  return _indices;
}
//---------------------------------------------------------------------------
template <typename T>
int MeshTags<T>::dim() const
{
  return _dim;
}
//---------------------------------------------------------------------------
template <typename T>
std::shared_ptr<const Mesh> MeshTags<T>::mesh() const
{
  return _mesh;
}
//---------------------------------------------------------------------------
template <typename T>
void MeshTags<T>::sort()
{
  // Prepare the sorting permutation
  std::vector<int> perm(_indices.size());
  std::iota(perm.begin(), perm.end(), 0);

  // Swap into a temporaries
  std::vector<std::int32_t> indices;
  indices.swap(_indices);
  std::vector<T> values;
  values.swap(_values);

  std::sort(perm.begin(), perm.end(), [&indices](const int a, const int b) {
    return (indices[a] < indices[b]);
  });

  // Make sure vectors are empty and preallocate space
  _indices.clear();
  _values.clear();
  _indices.reserve(indices.size());
  _values.reserve(values.size());

  // Apply sorting and insert
  for (std::size_t i = 0; i < indices.size(); ++i)
  {
    _indices.push_back(indices[perm[i]]);
    _values.push_back(values[perm[i]]);
  }
}
//---------------------------------------------------------------------------
template <typename T>
void MeshTags<T>::remove_duplicates()
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
//---------------------------------------------------------------------------
template <typename T>
mesh::MeshTags<T>
create_meshtags(MPI_Comm comm, const std::shared_ptr<const mesh::Mesh>& mesh,
                const mesh::CellType& entity_cell_type,
                const Eigen::Array<std::int64_t, Eigen::Dynamic, Eigen::Dynamic,
                                   Eigen::RowMajor>& topology,
                std::vector<T>& values)
{
  if ((std::size_t)topology.rows() != values.size())
    throw std::runtime_error("Number of entities and values must match");

  // Copy topology array into flattened vector
  std::vector<std::int64_t> topo_unique(
      topology.data(), topology.data() + topology.rows() * topology.cols());

  std::sort(topo_unique.begin(), topo_unique.end());
  topo_unique.erase(std::unique(topo_unique.begin(), topo_unique.end()),
                    topo_unique.end());

  const int e_dim = mesh::cell_dim(entity_cell_type);

  const int dim = mesh->topology().dim();
  auto e_to_v = mesh->topology().connectivity(e_dim, 0);
  assert(e_to_v);
  auto e_to_c = mesh->topology().connectivity(e_dim, dim);
  assert(e_to_c);
  auto c_to_v = mesh->topology().connectivity(dim, 0);
  assert(c_to_v);

  const std::vector<std::int64_t>& igi
      = mesh->geometry().input_global_indices();

  //
  // Send input global indices to process responsible for it, based on input
  // global index value
  //

  const std::int64_t num_igi_global = MPI::sum(comm, (std::int64_t)igi.size());

  // Split global array size and retrieve a range that this process/officer is
  // responsible for
  std::array<std::int64_t, 2> range = MPI::local_range(comm, num_igi_global);
  const int local_size = range[1] - range[0];

  const int comm_size = MPI::size(comm);
  std::vector<std::vector<std::int64_t>> send_igi(comm_size);
  std::vector<std::vector<std::int64_t>> recv_igi(comm_size);

  for (const std::int64_t& gi : igi)
  {
    // TODO: Optimise this call
    // Figure out which process responsible for the input global index
    const int officer = MPI::index_owner(comm_size, gi, num_igi_global);
    send_igi[officer].push_back(gi);
  }

  MPI::all_to_all(comm, send_igi, recv_igi);

  //
  // Handle received input global indices, i.e. put the owner of it to
  // a global position, which is its value
  //

  std::vector<std::int32_t> owners(local_size, 0);
  const std::size_t offset = MPI::global_offset(comm, local_size, true);

  for (std::int32_t i = 0; i < comm_size; ++i)
  {
    const std::int32_t num_recv_igi = (std::int32_t)recv_igi[i].size();
    for (std::int32_t j = 0; j < num_recv_igi; ++j)
    {
      const std::int32_t local_index = recv_igi[i][j] - offset;
      assert(local_size > local_index);
      assert(local_index >= 0);
      owners[local_index] = i;
    }
  }

  //
  // Distribute the owners of input global indices
  //

  Eigen::Map<Eigen::Array<std::int32_t, Eigen::Dynamic, 1>> owners_arr(
      owners.data(), owners.size());

  // Distribute owners and fetch owners for the input global indices read from
  // file, i.e. for the unique topology data in file
  const Eigen::Array<std::int32_t, Eigen::Dynamic, 1> dist_owners_arr
      = graph::Partitioning::distribute_data<std::int32_t>(comm, topo_unique,
                                                           owners_arr);

  //
  // Figure out which process needs input global indices read from file
  // and send to it
  //

  // Prepare an array where on n-th position is the owner of n-th node
  std::unordered_map<std::int64_t, int> topo_owners;
  for (std::size_t i = 0; i < topo_unique.size(); ++i)
    topo_owners[topo_unique[i]] = dist_owners_arr(i, 0);

  std::vector<std::vector<std::int64_t>> send_ents(comm_size);
  std::vector<std::vector<std::int64_t>> recv_ents(comm_size);
  std::vector<std::vector<T>> send_vals(comm_size);
  std::vector<std::vector<T>> recv_vals(comm_size);

  const int nnodes_per_entity = topology.cols();

  for (Eigen::Index e = 0; e < topology.rows(); ++e)
  {
    std::vector<std::int64_t> entity(nnodes_per_entity);
    std::vector<bool> sent(comm_size, false);

    for (int i = 0; i < nnodes_per_entity; ++i)
      entity[i] = topology(e, i);

    for (int i = 0; i < nnodes_per_entity; ++i)
    {
      // Entity could have as many owners as there are owners
      // of its nodes
      const int send_to = topo_owners[entity[i]];
      assert(send_to >= 0);
      if (!sent[send_to])
      {
        send_ents[send_to].insert(send_ents[send_to].end(), entity.begin(),
                                  entity.end());
        send_vals[send_to].push_back(values[e]);
        sent[send_to] = true;
      }
    }
  }

  MPI::all_to_all(comm, send_ents, recv_ents);
  MPI::all_to_all(comm, send_vals, recv_vals);

  //
  // Using just the information on current local mesh partition
  // prepare a mapping from *ordered* nodes of entity input global indices to
  // entity local index
  //

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

    // Iterate over all entities of the mesh
    // Find cell attached to the entity
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

      // Insert input global index for the node of the entitity
      entity_igi[v] = igi[cell_nodes[local_cell_vertex]];
    }

    // Sorting is needed to match with entities stored in file
    std::sort(entity_igi.begin(), entity_igi.end());
    entities_igi.insert({entity_igi, e});
  }

  //
  // Iterate over all received entities and find it in entities of
  // the mesh
  //

  std::vector<std::int32_t> indices;
  values.clear();

  for (std::int32_t i = 0; i < comm_size; ++i)
  {
    const std::int32_t num_recv_ents
        = (std::int32_t)(recv_ents[i].size() / nnodes_per_entity);
    for (std::int32_t e = 0; e < num_recv_ents; ++e)
    {
      std::vector<std::int64_t> _entity(&recv_ents[i][nnodes_per_entity * e],
                                        &recv_ents[i][nnodes_per_entity * e]
                                            + nnodes_per_entity);

      std::sort(_entity.begin(), _entity.end());

      const auto it = entities_igi.find(_entity);
      if (it != entities_igi.end())
      {
        indices.push_back(it->second);
        values.push_back(recv_vals[i][e]);
      }
    }
  }

  return mesh::MeshTags<T>(mesh, e_dim, indices, values);
}
} // namespace mesh
} // namespace dolfinx
