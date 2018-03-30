// Copyright (C) 2011-2013 Garth N. Wells
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include <array>
#include <dolfin/common/MPI.h>
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
  static std::
      unordered_map<std::uint32_t,
                    std::vector<std::pair<std::uint32_t, std::uint32_t>>>
      compute_shared_entities(const Mesh& mesh, std::size_t d);

  /// Reorders the vertices in a distributed mesh according to
  /// their global index, and redistributes them evenly across processes
  /// returning the coordinates as a local vector
  static std::vector<double>
  reorder_vertices_by_global_indices(const Mesh& mesh);

  /// Reorder the values (of given width) in data to be in global vertex
  /// index order on the Mesh, redistributing evenly across processes
  static void reorder_values_by_global_indices(const Mesh& mesh,
                                               std::vector<double>& data,
                                               const std::size_t width);

  /// Reorder the values of given width, according to explicit global
  /// indices, distributing evenly across processes
  static void reorder_values_by_global_indices(
      MPI_Comm mpi_comm, std::vector<double>& values, const std::size_t width,
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
};
}
}