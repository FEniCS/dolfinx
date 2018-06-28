// Copyright (C) 2011-2013 Garth N. Wells
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include <array>
#include <complex>
#include <dolfin/common/MPI.h>
#include <dolfin/common/Timer.h>
#include <dolfin/common/types.h>
#include <map>
#include <numeric>
#include <set>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

namespace dolfin
{

namespace mesh
{
class Mesh;

/// This class provides various functionality for working with
/// distributed meshes.

class DistributedMeshTools
{
public:
  /// Create global entity indices for entities of dimension d
  static void number_entities(const Mesh& mesh, std::size_t d);

  /// Create global entity indices for entities of dimension d for
  /// given global vertex indices. Returns  global_entity_indices,
  /// shared_entities, and XXXX?
  static std::tuple<std::vector<std::int64_t>,
                    std::map<std::int32_t, std::set<std::uint32_t>>,
                    std::size_t>
  number_entities(
      const Mesh& mesh,
      const std::map<std::uint32_t, std::pair<std::uint32_t, std::uint32_t>>&
          slave_entities,
      std::size_t d);

  /// Compute number of cells connected to each facet
  /// (globally). Facets on internal boundaries will be connected to
  /// two cells (with the cells residing on neighboring processes)
  static void init_facet_cell_connections(Mesh& mesh);

  /// Find processes that own or share mesh entities (using entity
  /// global indices). Returns (global_dof, set(process_num,
  /// local_index)). Exclusively local entities will not appear in
  /// the map. Works only for vertices and cells
  static std::map<std::size_t, std::set<std::pair<std::size_t, std::size_t>>>
  locate_off_process_entities(const std::vector<std::size_t>& entity_indices,
                              std::size_t dim, const Mesh& mesh);

  /// Compute map from local index of shared entity to list
  /// of sharing process and local index,
  /// i.e. (local index, [(sharing process p, local index on p)])
  static std::unordered_map<
      std::uint32_t, std::vector<std::pair<std::uint32_t, std::uint32_t>>>
  compute_shared_entities(const Mesh& mesh, std::size_t d);

  /// Reorders the points in a distributed mesh according to
  /// their global index, and redistributes them evenly across processes
  /// returning the coordinates as a local vector
  /// @param mesh
  ///    a Mesh
  /// @return EigenRowArrayXXd
  ///    Array of points in global order
  static EigenRowArrayXXd reorder_points_by_global_indices(const Mesh& mesh);

  /// Reorder the values according to explicit global indices, distributing
  /// evenly across processes
  /// @param mpi_comm
  ///    MPI Communicator
  /// @param values
  ///    Values to reorder
  /// @param global_indices
  ///    Global index for each row of values
  static EigenRowArrayXXd reorder_values_by_global_indices(
      MPI_Comm mpi_comm, const Eigen::Ref<const EigenRowArrayXXd>& values,
      const std::vector<std::int64_t>& global_indices);

  /// Reorder the values according to explicit global indices, distributing
  /// evenly across processes
  /// @param mpi_comm
  ///    MPI Communicator
  /// @param values
  ///    Complex values to reorder
  /// @param global_indices
  ///    Global index for each row of values
  static Eigen::Array<std::complex<double>, Eigen::Dynamic, Eigen::Dynamic,
                      Eigen::RowMajor>
  reorder_values_by_global_indices(
      MPI_Comm mpi_comm,
      const Eigen::Ref<const Eigen::Array<std::complex<double>, Eigen::Dynamic,
                                          Eigen::Dynamic, Eigen::RowMajor>>&
          values,
      const std::vector<std::int64_t>& global_indices);

private:
  // Data structure for a mesh entity (list of vertices, using
  // global indices)
  typedef std::vector<std::size_t> Entity;

  // Data structure for mesh entity data
  struct EntityData
  {
    // Constructor
    EntityData() : local_index(0) {}

    // Move constructor
    EntityData(EntityData&&) = default;

    // Move assignment
    EntityData& operator=(EntityData&&) = default;

    // Constructor  (index is local)
    explicit EntityData(std::uint32_t index) : local_index(index) {}

    // Constructor (index is local)
    EntityData(std::uint32_t index, const std::vector<std::uint32_t>& procs)
        : local_index(index), processes(procs)
    {
      // Do nothing
    }

    // Constructor  (index is local)
    EntityData(std::uint32_t index, std::uint32_t process)
        : local_index(index), processes(1, process)
    {
      // Do nothing
    }

    // Local (this process) entity index
    std::uint32_t local_index;

    // Processes on which entity resides
    std::vector<std::uint32_t> processes;
  };

  // Compute ownership of entities ([entity vertices], data)
  //  [0]: owned exclusively (will be numbered by this process)
  //  [1]: owned and shared (will be numbered by this process, and number
  //       communicated to other processes)
  //  [2]: not owned but shared (will be numbered by another process,
  //       and number communicated to this processes)
  //  Returns (owned_entities,  shared_entities)
  static std::pair<std::vector<std::size_t>,
                   std::array<std::map<Entity, EntityData>, 2>>
  compute_entity_ownership(
      const MPI_Comm mpi_comm,
      const std::map<std::vector<std::size_t>, std::uint32_t>& entities,
      const std::map<std::int32_t, std::set<std::uint32_t>>&
          shared_vertices_local,
      const std::vector<std::int64_t>& global_vertex_indices, std::size_t d);

  // Build preliminary 'guess' of shared entities. This function does
  // not involve any inter-process communication. Returns (owned_entities,
  // entity_ownership);
  static std::pair<std::vector<std::size_t>,
                   std::array<std::map<Entity, EntityData>, 2>>
  compute_preliminary_entity_ownership(
      const MPI_Comm mpi_comm,
      const std::map<std::size_t, std::set<std::uint32_t>>& shared_vertices,
      const std::map<Entity, std::uint32_t>& entities);

  // Communicate with other processes to finalise entity ownership
  static void compute_final_entity_ownership(
      const MPI_Comm mpi_comm, std::vector<std::size_t>& owned_entities,
      std::array<std::map<Entity, EntityData>, 2>& entity_ownership);

  // Check if all entity vertices are the shared vertices in overlap
  static bool is_shared(
      const std::vector<std::size_t>& entity_vertices,
      const std::map<std::size_t, std::set<std::uint32_t>>& shared_vertices);

  // Compute and return (number of global entities, process offset)
  static std::pair<std::size_t, std::size_t> compute_num_global_entities(
      const MPI_Comm mpi_comm, std::size_t num_local_entities,
      std::size_t num_processes, std::size_t process_number);

  /// Reorder the values according to explicit global indices, distributing
  /// evenly across processes
  /// @param mpi_comm
  ///    MPI Communicator
  /// @param values
  ///    Values to reorder
  /// @param global_indices
  ///    Global index for each row of values
  template <typename Scalar>
  static Eigen::Array<Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
  reorder_values(
      MPI_Comm mpi_comm,
      const Eigen::Ref<const Eigen::Array<
          Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>& values,
      const std::vector<std::int64_t>& global_indices)
  {
    common::Timer t("DistributedMeshTools: reorder values");

    // Number of items to redistribute
    const std::size_t num_local_indices = global_indices.size();
    assert(num_local_indices == (std::size_t)values.rows());

    // Calculate size of overall global vector by finding max index value
    // anywhere
    const std::size_t global_vector_size
        = MPI::max(mpi_comm, *std::max_element(global_indices.begin(),
                                               global_indices.end()))
          + 1;

    // Send unwanted values off process
    const std::size_t mpi_size = MPI::size(mpi_comm);
    std::vector<std::vector<std::size_t>> indices_to_send(mpi_size);
    std::vector<std::vector<Scalar>> values_to_send(mpi_size);

    // Go through local vector and append value to the appropriate list
    // to send to correct process
    for (std::size_t i = 0; i != num_local_indices; ++i)
    {
      const std::size_t global_i = global_indices[i];
      const std::size_t process_i
          = MPI::index_owner(mpi_comm, global_i, global_vector_size);
      indices_to_send[process_i].push_back(global_i);
      values_to_send[process_i].insert(values_to_send[process_i].end(),
                                       values.row(i).data(),
                                       values.row(i).data() + values.cols());
    }

    // Redistribute the values to the appropriate process - including
    // self. All values are "in the air" at this point. Receive into flat
    // arrays.
    std::vector<std::size_t> received_indices;
    std::vector<Scalar> received_values;
    MPI::all_to_all(mpi_comm, indices_to_send, received_indices);
    MPI::all_to_all(mpi_comm, values_to_send, received_values);

    // Map over received values as Eigen array
    assert(received_indices.size() * values.cols() == received_values.size());
    Eigen::Map<
        Eigen::Array<Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
        received_values_array(received_values.data(), received_indices.size(),
                              values.cols());

    // Create array for new data. Note that any indices which are not received
    // will be uninitialised.
    const std::array<std::int64_t, 2> range
        = MPI::local_range(mpi_comm, global_vector_size);
    Eigen::Array<Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
        new_values(range[1] - range[0], values.cols());

    // Go through received data in descending order, and place in local
    // partition of the global vector. Any duplicate data (with same index)
    // will be overwritten by values from the lowest rank process.
    for (std::int32_t j = received_indices.size() - 1; j >= 0; --j)
    {
      const std::int64_t global_i = received_indices[j];
      assert(global_i >= range[0] && global_i < range[1]);
      new_values.row(global_i - range[0]) = received_values_array.row(j);
    }

    return new_values;
  }
};
} // namespace mesh
} // namespace dolfin
