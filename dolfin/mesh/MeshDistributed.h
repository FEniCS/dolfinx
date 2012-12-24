// Copyright (C) 2011 Garth N. Wells
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
// Last changed:

#ifndef __MESH_DISTRIBUTED_H
#define __MESH_DISTRIBUTED_H

#include <map>
#include <set>
#include <utility>
#include <vector>

namespace dolfin
{

  class Mesh;

  /// This class provides various funtionality for working with
  /// distributed meshes.

  class MeshDistributed
  {
  public:

    /// Create global entity indices for entities of dimension d
    static void number_entities(const Mesh& mesh, std::size_t d);

    // Compute number of cells connected to each facet (globally). Facets
    // on internal boundaries will be connected to two cells (with the
    // cells residing on neighboring processes)
    static void init_facet_cell_connections(Mesh& mesh);

    /// Find processes that own or share mesh entities (using
    /// entity global indices). Returns
    /// (global_dof, set(process_num, local_index)). Exclusively local
    /// entities will not appear in the map. Works only for vertices and
    /// cells
    static std::map<std::size_t, std::set<std::pair<std::size_t, std::size_t> > >
    locate_off_process_entities(const std::vector<std::size_t>& entity_indices,
                              std::size_t dim, const Mesh& mesh);

    /// Compute map from local index of shared entity to list
    /// of sharing process and local index,
    /// i.e. (local index my process, [sharing process p, local index on p])
    static std::map<std::size_t, std::vector<std::pair<std::size_t, std::size_t> > >
      compute_shared_entities(const Mesh& mesh, std::size_t d);

  private:

    // Data structure for a mesh entity (list of vertices, using global indices)
    typedef std::vector<std::size_t> Entity;

    // Data structure to mesh entity data
    struct EntityData
    {
      // Constructor
      EntityData() : local_index(0) {}

      // Constructor
      explicit EntityData(std::size_t local_index) : local_index(local_index) {}

      // Constructor
      EntityData(std::size_t local_index, const std::vector<std::size_t>& processes)
        : local_index(local_index), processes(processes) {}

      // Local (this process) entity index
      std::size_t local_index;

      // Processes on which entity resides
      std::vector<std::size_t> processes;
    };

    // Compute ownership of entities ([entity vertices], data)
    //  [0]: owned exclusively (will be numbered by this process)
    //  [1]: owned and shared (will be numbered by this process, and number
    //       commuicated to other processes)
    //  [2]: not owned but shared (will be numbered by another process,
    //       and number commuicated to this processes)
    static boost::array<std::map<Entity, EntityData>, 3>
          compute_entity_ownership(const Mesh& mesh, std::size_t d);

    // Build preliminary 'guess' of shared enties
    static void compute_preliminary_entity_ownership(
          const std::map<std::size_t, std::set<std::size_t> >& shared_vertices,
          const std::map<Entity, std::size_t>& entities,
          boost::array<std::map<Entity, EntityData>, 3>& entity_ownership);

    // Communicate with other processes to finalise entity ownership
    static void compute_final_entity_ownership(boost::array<std::map<Entity, EntityData>, 3>& entity_ownership);

    // Check if all entity vertices are the shared vertices in overlap
    static bool is_shared(const std::vector<std::size_t>& entity_vertices,
               const std::map<std::size_t, std::set<std::size_t> >& shared_vertices);

    // Compute and return (number of global entities, process offset)
    static std::pair<std::size_t, std::size_t>
      compute_num_global_entities(std::size_t num_local_entities,
                                  std::size_t num_processes,
                                  std::size_t process_number);

  };

}

#endif
