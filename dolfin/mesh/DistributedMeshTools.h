// Copyright (C) 2011-2013 Garth N. Wells
//
// This file is part of DOLFIN.
//
// DOLFIN is free software: you can redistribute it and/or modify
// it under the terms of the GNU Lesser General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// DOLFIN is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with DOLFIN. If not, see <http://www.gnu.org/licenses/>.
//
// First added:  2011-09-17
// Last changed: 2013-01-29

#ifndef __MESH_DISTRIBUTED_TOOLS_H
#define __MESH_DISTRIBUTED_TOOLS_H

#include <array>
#include <map>
#include <numeric>
#include <set>
#include <utility>
#include <unordered_map>
#include <vector>
#include <dolfin/common/MPI.h>

namespace dolfin
{

  class Mesh;

  /// Functionality for working with distributed meshes.

  class DistributedMeshTools
  {
  public:

    /// Create global entity indices for entities of dimension d
    static void number_entities(const Mesh& mesh, std::size_t d);

    /// Create global entity indices for entities of dimension d for
    /// given global vertex indices.
    static std::size_t number_entities(
      const Mesh& mesh,
      const std::map<unsigned int,
      std::pair<unsigned int, unsigned int>>& slave_entities,
      std::vector<std::size_t>& global_entity_indices,
      std::map<std::int32_t, std::set<unsigned int>>& shared_entities,
      std::size_t d);

    /// Compute number of cells connected to each facet
    /// (globally). Facets on internal boundaries will be connected to
    /// two cells (with the cells residing on neighboring processes)
    static void init_facet_cell_connections(Mesh& mesh);

    /// Find processes that own or share mesh entities (using entity
    /// global indices). Returns (global_dof, set(process_num,
    /// local_index)). Exclusively local entities will not appear in
    /// the map. Works only for vertices and cells
    static
      std::map<std::size_t, std::set<std::pair<std::size_t, std::size_t>>>
      locate_off_process_entities(const std::vector<std::size_t>&
                                  entity_indices,
                                  std::size_t dim, const Mesh& mesh);

    /// Compute map from local index of shared entity to list
    /// of sharing process and local index,
    /// i.e. (local index, [(sharing process p, local index on p)])
    static std::unordered_map<unsigned int,
      std::vector<std::pair<unsigned int, unsigned int>>>
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
    static void reorder_values_by_global_indices(MPI_Comm mpi_comm,
                          std::vector<double>& values,
                          const std::size_t width,
                          const std::vector<std::size_t>& global_indices);

  private:

    // Data structure for a mesh entity (list of vertices, using
    // global indices)
    typedef std::vector<std::size_t> Entity;

    // Data structure to mesh entity data
    struct EntityData
    {
      // Constructor
      EntityData() : local_index(0) {}

      // Constructor  (index is local)
      explicit EntityData(unsigned int index) : local_index(index) {}

      // Constructor (index is local)
      EntityData(unsigned int index, const std::vector<unsigned int>& procs)
        : local_index(index), processes(procs) {}

      // Constructor  (index is local)
      EntityData(unsigned int index, unsigned int process)
        : local_index(index), processes(1, process) {}

      // Local (this process) entity index
      unsigned int local_index;

      // Processes on which entity resides
      std::vector<unsigned int> processes;
    };

    // Compute ownership of entities ([entity vertices], data)
    //  [0]: owned exclusively (will be numbered by this process)
    //  [1]: owned and shared (will be numbered by this process, and number
    //       communicated to other processes)
    //  [2]: not owned but shared (will be numbered by another process,
    //       and number communicated to this processes)
    static void compute_entity_ownership(
      const MPI_Comm mpi_comm,
      const std::map<std::vector<std::size_t>, unsigned int>& entities,
      const std::map<std::int32_t, std::set<unsigned int> >& shared_vertices_local,
      const std::vector<std::size_t>& global_vertex_indices,
      std::size_t d,
      std::vector<std::size_t>& owned_entities,
      std::array<std::map<Entity, EntityData>, 2>& shared_entities);

    // Build preliminary 'guess' of shared entities. This function does
    // not involve any inter-process communication.
    static void compute_preliminary_entity_ownership(
      const MPI_Comm mpi_comm,
      const std::map<std::size_t, std::set<unsigned int> >& shared_vertices,
      const std::map<Entity, unsigned int>& entities,
      std::vector<std::size_t>& owned_entities,
      std::array<std::map<Entity, EntityData>, 2>& entity_ownership);

    // Communicate with other processes to finalise entity ownership
    static void
      compute_final_entity_ownership(const MPI_Comm mpi_comm,
                                     std::vector<std::size_t>& owned_entities,
                                     std::array<std::map<Entity,
                                     EntityData>, 2>& entity_ownership);

    // Check if all entity vertices are the shared vertices in overlap
    static bool is_shared(const std::vector<std::size_t>& entity_vertices,
                const std::map<std::size_t, std::set<unsigned int> >& shared_vertices);

    // Compute and return (number of global entities, process offset)
    static std::pair<std::size_t, std::size_t>
      compute_num_global_entities(const MPI_Comm mpi_comm,
                                  std::size_t num_local_entities,
                                  std::size_t num_processes,
                                  std::size_t process_number);

  };

}

#endif
