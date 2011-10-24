// Copyright (C) 2008-2009 Niclas Jansson, Ola Skavhaug and Anders Logg
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
// Modified by Garth N. Wells, 2010
// Modified by Kent-Andre Mardal, 2011
//
// First added:  2008-12-01
// Last changed: 2010-04-04

#ifndef __MESH_PARTITIONING_H
#define __MESH_PARTITIONING_H

#include <map>
#include <utility>
#include <vector>
#include <dolfin/common/types.h>
#include <dolfin/log/log.h>
#include "LocalMeshValueCollection.h"
#include "Mesh.h"
#include "MeshDistributed.h"
#include "ParallelData.h"

namespace dolfin
{
  // Note: MeshFunction and MeshValueCollection cannot apear in the
  // implementations that appear in this file of the templated functions
  // as this leads to a circular dependency. Therefore the functions are
  // templated over these types.

  template <typename T> class MeshFunction;
  template <typename T> class MeshValueCollection;
  class LocalMeshData;

  /// This class partitions and distributes a mesh based on
  /// partitioned local mesh data. Note that the local mesh data will
  /// also be repartitioned and redistributed during the computation
  /// of the mesh partitioning.
  ///
  /// After partitioning, each process has a local mesh and set of
  /// mesh data that couples the meshes together.
  ///
  /// The following mesh data is created:
  ///
  /// 1. "global entity indices 0" (MeshFunction<uint>)
  ///
  /// This maps each local vertex to its global index.
  ///
  /// 2. "overlap" (std::map<uint, std::vector<uint> >)
  ///
  /// This maps each shared vertex to a list of the processes sharing
  /// the vertex.
  ///
  /// 3. "global entity indices %d" (MeshFunction<uint>)
  ///
  /// After partitioning, the function number_entities() may be called
  /// to create global indices for all entities of a given topological
  /// dimension. These are stored as mesh data (MeshFunction<uint>)
  /// named
  ///
  ///    "global entity indices 1"
  ///    "global entity indices 2"
  ///    etc
  ///
  /// 4. "num global entities" (std::vector<uint>)
  ///
  /// The function number_entities also records the number of global
  /// entities for the dimension of the numbered entities in the array
  /// named "num global entities". This array has size D + 1, where D
  /// is the topological dimension of the mesh. This array is
  /// initially created by the mesh and then contains only the number
  /// entities of dimension 0 (vertices) and dimension D (cells).

  class MeshPartitioning
  {
  public:

   /// Build a partitioned mesh based on local meshes
    static void build_distributed_mesh(Mesh& mesh);

    /// Build a partitioned mesh based on local mesh data
    static void build_distributed_mesh(Mesh& mesh, LocalMeshData& data);

    template<typename T>
    static void build_distributed_value_collection(MeshValueCollection<T>& values,
               const LocalMeshValueCollection<T>& local_data, const Mesh& mesh);

    /// Create global entity indices for entities of dimension d
    static void number_entities(const Mesh& mesh, uint d);

  private:

    /// Create a partitioned mesh based on local mesh data
    static void partition(Mesh& mesh, LocalMeshData& data);

    /// Create and attach distributed MeshDomains from local_data
    static void build_mesh_domains(Mesh& mesh, const LocalMeshData& local_data);

    /// Create and attach distributed MeshDomains from local_data
    /// [entry, (cell_index, local_index, value)]
    template<typename T, typename MeshValueCollection>
    static void build_mesh_value_collection(const Mesh& mesh,
      const std::vector<std::pair<std::pair<uint, uint>, T> >& local_value_data,
      MeshValueCollection& mesh_values);

    // Compute and return (number of global entities, process offset)
    static std::pair<uint, uint> compute_num_global_entities(uint num_local_entities,
                                                     uint num_processes,
                                                     uint process_number);

    // Build preliminary 'guess' of shared enties
    static void compute_preliminary_entity_ownership(const std::map<std::vector<uint>, uint>& entities,
          const std::map<uint, std::vector<uint> >& shared_vertices,
          std::map<std::vector<uint>, uint>& owned_entity_indices,
          std::map<std::vector<uint>, uint>& shared_entity_indices,
          std::map<std::vector<uint>, std::vector<uint> >& shared_entity_processes,
          std::map<std::vector<uint>, uint>& ignored_entity_indices,
          std::map<std::vector<uint>, std::vector<uint> >& ignored_entity_processes);

    // Communicate with other processes to finalise entity ownership
    static void compute_final_entity_ownership(std::map<std::vector<uint>, uint>& owned_entity_indices,
          std::map<std::vector<uint>, uint>& shared_entity_indices,
          std::map<std::vector<uint>, std::vector<uint> >& shared_entity_processes,
          std::map<std::vector<uint>, uint>& ignored_entity_indices,
          std::map<std::vector<uint>, std::vector<uint> >& ignored_entity_processes);

    // Distribute cells
    static void distribute_cells(LocalMeshData& data,
                                 const std::vector<uint>& cell_partition);

    // Distribute vertices
    static void distribute_vertices(LocalMeshData& data,
                                    std::map<uint, uint>& glob2loc);

    // Build mesh
    static void build_mesh(Mesh& mesh, const LocalMeshData& data,
                           std::map<uint, uint>& glob2loc);

    // Check if all entity vertices are in overlap
    static bool in_overlap(const std::vector<uint>& entity_vertices,
                           const std::map<uint, std::vector<uint> >& overlap);

    // Mark non-shared mesh entities
    static void mark_nonshared(const std::map<std::vector<uint>, uint>& entities,
               const std::map<std::vector<uint>, uint>& shared_entity_indices,
               const std::map<std::vector<uint>, uint>& ignored_entity_indices,
               MeshFunction<bool>& exterior_facets);
  };

  //---------------------------------------------------------------------------
  template<typename T>
  void MeshPartitioning::build_distributed_value_collection(MeshValueCollection<T>& values,
             const LocalMeshValueCollection<T>& local_data, const Mesh& mesh)
  {
    // Extract data
    const std::vector<std::pair<std::pair<uint, uint>, T> >& local_values
      = local_data.values();

    // Build MeshValueCollection from local data
    build_mesh_value_collection(mesh, local_values, values);
  }
  //---------------------------------------------------------------------------
  template<typename T, typename MeshValueCollection>
  void MeshPartitioning::build_mesh_value_collection(const Mesh& mesh,
    const std::vector<std::pair<std::pair<uint, uint>, T> >& local_value_data,
    MeshValueCollection& mesh_values)
  {
    // Get topological dimensions
    const uint D = mesh.topology().dim();
    const uint dim = mesh_values.dim();

    // Clear MeshValueCollection values
    mesh_values.values().clear();

    // Initialise global entity numbering
    MeshPartitioning::number_entities(mesh, dim);
    MeshPartitioning::number_entities(mesh, D);

    if (dim == 0)
      error("MeshPartitioning::build_mesh_value_collection needs updating for vertices.");

    // Get mesh value collection used for marking
    MeshValueCollection& markers = mesh_values;

    // Get local mesh data for domains
    const std::vector< std::pair<std::pair<uint, uint>, T> >&
      ldata = local_value_data;

    // Get local local-to-global map
    if (!mesh.parallel_data().have_global_entity_indices(D))
      error("Do not have have_global_entity_indices");

    // Get global indices on local process
    const std::vector<uint> global_entity_indices
      = mesh.parallel_data().global_entity_indices_as_vector(D);

    // Add local (to this process) data to domain marker
    std::vector<uint>::iterator it;
    std::vector<uint> off_process_global_cell_entities;
    for (uint i = 0; i < ldata.size(); ++i)
    {
      const uint global_cell_index = ldata[i].first.first;
      std::vector<uint>::const_iterator it;
      it = std::find(global_entity_indices.begin(), global_entity_indices.end(), global_cell_index);
      if (it != global_entity_indices.end())
      {
        const uint local_cell_index = std::distance(global_entity_indices.begin(), it);
        const uint entity_local_index = ldata[i].first.second;
        const T value = ldata[i].second;
        markers.set_value(local_cell_index, entity_local_index, value);
      }
      else
        off_process_global_cell_entities.push_back(global_cell_index);
    }

    // Get destinations and local cell index at destination for off-process cells
    const std::map<uint, std::set<std::pair<uint, uint> > >
      entity_hosts = MeshDistributed::off_process_indices(off_process_global_cell_entities, D, mesh);

    // Pack data to send to appropriate process
    std::vector<uint> send_data0;
    std::vector<T> send_data1;
    std::vector<uint> destinations0;
    std::vector<uint> destinations1;
    std::map<uint, std::set<std::pair<uint, uint> > >::const_iterator entity_host;
    for (entity_host = entity_hosts.begin(); entity_host != entity_hosts.end(); ++entity_host)
    {
      const uint host_global_cell_index = entity_host->first;
      const std::set<std::pair<uint, uint> >& processes_data = entity_host->second;

      // Loop over local data
      for (uint i = 0; i < ldata.size(); ++i)
      {
        const uint local_global_cell_index = ldata[i].first.first;
        if (local_global_cell_index == host_global_cell_index)
        {
          const uint local_entity_index = ldata[i].first.second;
          const T domain_value = ldata[i].second;

          std::set<std::pair<uint, uint> >::const_iterator process_data;
          for (process_data = processes_data.begin(); process_data != processes_data.end(); ++process_data)
          {
            const uint proc = process_data->first;
            const uint local_cell_entity = process_data->second;

            send_data0.push_back(local_cell_entity);
            send_data0.push_back(local_entity_index);
            destinations0.insert(destinations0.end(), 2, proc);

            send_data1.push_back(domain_value);
            destinations1.push_back(proc);
          }
        }
      }
    }

    // Send/receive data
    std::vector<uint> received_data0;
    std::vector<T> received_data1;
    MPI::distribute(send_data0, destinations0, received_data0);
    MPI::distribute(send_data1, destinations1, received_data1);
    assert(2*received_data1.size() == received_data0.size());

    // Add received data to mesh domain
    for (uint i = 0; i < received_data1.size(); ++i)
    {
      const uint local_cell_entity = received_data0[2*i];
      const uint local_entity_index = received_data0[2*i + 1];
      const T value = received_data1[i];
      assert(local_cell_entity < mesh.num_cells());
      markers.set_value(local_cell_entity, local_entity_index, value);
    }
  }
  //---------------------------------------------------------------------------

}

#endif
