// Copyright (C) 2008-2012 Niclas Jansson, Ola Skavhaug, Anders Logg and
// Garth N. Wells
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
// Last changed: 2012-12-24

#ifndef __MESH_PARTITIONING_H
#define __MESH_PARTITIONING_H

#include <map>
#include <utility>
#include <vector>
#include <boost/multi_array.hpp>
#include <dolfin/log/log.h>
#include "DistributedMeshTools.h"
#include "LocalMeshValueCollection.h"
#include "Mesh.h"

namespace dolfin
{
  // Developer note: MeshFunction and MeshValueCollection cannot appear
  // in the implementations that appear in this file of the templated
  // functions as this leads to a circular dependency. Therefore the
  // functions are templated over these types.

  template <typename T> class MeshFunction;
  template <typename T> class MeshValueCollection;
  class LocalMeshData;

  /// This class partitions and distributes a mesh based on
  /// partitioned local mesh data.The local mesh data will
  /// also be repartitioned and redistributed during the computation
  /// of the mesh partitioning.
  ///
  /// After partitioning, each process has a local mesh and some data
  /// that couples the meshes together.

  class MeshPartitioning
  {
  public:

    /// Build a partitioned mesh based from a local mesh on process 0
    static void build_distributed_mesh(Mesh& mesh);

    /// Build a partitioned mesh based from a local mesh on process 0
    /// with supplied destination processes for each cell
    static void build_distributed_mesh(Mesh& mesh,
                                       const std::vector<std::size_t>& cell_partition);

    /// Build a partitioned mesh from local mesh data that is
    /// distributed across processes
    static void build_distributed_mesh(Mesh& mesh, const LocalMeshData& data);

    /// Build a MeshValueCollection based on LocalMeshValueCollection
    template<typename T>
    static void build_distributed_value_collection(MeshValueCollection<T>& values,
               const LocalMeshValueCollection<T>& local_data, const Mesh& mesh);

  private:

    // Compute cell partitioning for local mesh data. Returns
    // cell->process vector for cells in LocalMeshData
    static std::vector<std::size_t> partition_cells(Mesh& mesh,
                                                    const LocalMeshData& data);

    // Build mesh from local mesh data with a computed partition
    static void build(Mesh& mesh, const LocalMeshData& data,
                      const std::vector<std::size_t>& cell_partition);

    // This function takes the partition computed by the partitioner
    // (which tells us to which process each of the local cells stored in
    // LocalMeshData on this process belongs. We use MPI::distribute to
    // redistribute all cells (the global vertex indices of all cells).
    static void distribute_cells(const LocalMeshData& data,
                                 const std::vector<std::size_t>& cell_partition,
                                 std::vector<std::size_t>& cell_local_to_global_indices,
                                 boost::multi_array<std::size_t, 2>& cell_local_vertices);

    // Distribute vertices
    static void distribute_vertices(const LocalMeshData& data,
                  const boost::multi_array<std::size_t, 2>& cell_local_vertices,
                  std::vector<std::size_t>& vertex_local_to_global_indices,
                  std::map<std::size_t, std::size_t>& vertex_global_to_local_indices,
                  boost::multi_array<double, 2>& vertex_coordinates);

    // Build mesh
    static void build_mesh(Mesh& mesh,
                   const std::vector<std::size_t>& global_cell_indices,
                   const boost::multi_array<std::size_t, 2>& cell_vertices,
                   const std::vector<std::size_t>& vertex_indices,
                   const boost::multi_array<double, 2>& vertex_coordinates,
                   const std::map<std::size_t, std::size_t>& vertex_global_to_local_indices,
                   std::size_t tdim, std::size_t gdim, std::size_t num_global_cells,
                   std::size_t num_global_vertices);

    // Create and attach distributed MeshDomains from local_data
    static void build_mesh_domains(Mesh& mesh,
                                   const LocalMeshData& local_data);

    // Create and attach distributed MeshDomains from local_data
    // [entry, (cell_index, local_index, value)]
    template<typename T, typename MeshValueCollection>
    static void build_mesh_value_collection(const Mesh& mesh,
      const std::vector<std::pair<std::pair<std::size_t, std::size_t>, T> >& local_value_data,
      MeshValueCollection& mesh_values);
  };

  //---------------------------------------------------------------------------
  template<typename T>
  void MeshPartitioning::build_distributed_value_collection(MeshValueCollection<T>& values,
             const LocalMeshValueCollection<T>& local_data, const Mesh& mesh)
  {
    // Extract data
    const std::vector<std::pair<std::pair<std::size_t, std::size_t>, T> >& local_values
      = local_data.values();

    // Build MeshValueCollection from local data
    build_mesh_value_collection(mesh, local_values, values);
  }
  //---------------------------------------------------------------------------
  template<typename T, typename MeshValueCollection>
  void MeshPartitioning::build_mesh_value_collection(const Mesh& mesh,
    const std::vector<std::pair<std::pair<std::size_t, std::size_t>, T> >& local_value_data,
    MeshValueCollection& mesh_values)
  {
    // Get topological dimensions
    const std::size_t D = mesh.topology().dim();
    const std::size_t dim = mesh_values.dim();

    // Clear MeshValueCollection values
    mesh_values.clear();

    // Initialise global entity numbering
    DistributedMeshTools::number_entities(mesh, dim);

    if (dim == 0)
    {
      // MeshPartitioning::build_mesh_value_collection needs updating
      // for vertices
      dolfin_not_implemented();
    }

    // Get mesh value collection used for marking
    MeshValueCollection& markers = mesh_values;

    // Get local mesh data for domains
    const std::vector< std::pair<std::pair<std::size_t, std::size_t>, T> >&
      ldata = local_value_data;

    // Get local local-to-global map
    if (!mesh.topology().have_global_indices(D))
    {
      dolfin_error("MeshPartitioning.h",
                   "build mesh value collection",
                   "Do not have have_global_entity_indices");
    }

    // Get global indices on local process
    const std::vector<std::size_t> global_entity_indices
      = mesh.topology().global_indices(D);

    // Add local (to this process) data to domain marker
    std::vector<std::size_t> off_process_global_cell_entities;

    // Build and populate a local map for global_entity_indices
    std::map<std::size_t, std::size_t> map_of_global_entity_indices;
    for (std::size_t i = 0; i < global_entity_indices.size(); i++)
      map_of_global_entity_indices[global_entity_indices[i]] = i;

    for (std::size_t i = 0; i < ldata.size(); ++i)
    {
      const std::size_t global_cell_index = ldata[i].first.first;
      std::map<std::size_t, std::size_t>::const_iterator data
        = map_of_global_entity_indices.find(global_cell_index);
      if (data != map_of_global_entity_indices.end())
      {
        const std::size_t local_cell_index = data->second;
        const std::size_t entity_local_index = ldata[i].first.second;
        const T value = ldata[i].second;
        markers.set_value(local_cell_index, entity_local_index, value);
      }
      else
        off_process_global_cell_entities.push_back(global_cell_index);
    }

    // Get destinations and local cell index at destination for
    // off-process cells
    const std::map<std::size_t, std::set<std::pair<std::size_t, std::size_t> > >
      entity_hosts
      = DistributedMeshTools::locate_off_process_entities(off_process_global_cell_entities,
                                                          D, mesh);

    // Number of MPI processes
    const std::size_t num_processes = MPI::num_processes();

    // Pack data to send to appropriate process
    std::vector<std::vector<std::size_t> > send_data0(num_processes);
    std::vector<std::vector<T> > send_data1(num_processes);
    std::map<std::size_t, std::set<std::pair<std::size_t, std::size_t> > >::const_iterator entity_host;

    {
      // Build a convenience map in order to speedup the loop over
      // local data
      std::map<std::size_t, std::set<std::size_t> > map_of_ldata;
      for (std::size_t i = 0; i < ldata.size(); ++i)
        map_of_ldata[ldata[i].first.first].insert(i);

      for (entity_host = entity_hosts.begin(); entity_host != entity_hosts.end();
           ++entity_host)
      {
        const std::size_t host_global_cell_index = entity_host->first;
        const std::set<std::pair<std::size_t, std::size_t> >& processes_data
          = entity_host->second;

        // Loop over local data
        std::map<std::size_t, std::set<std::size_t> >::const_iterator ldata_it
          = map_of_ldata.find(host_global_cell_index);
        if (ldata_it != map_of_ldata.end())
        {
          for (std::set<std::size_t>::const_iterator it = ldata_it->second.begin();
               it != ldata_it->second.end(); it++)
          {
            const std::size_t local_entity_index = ldata[*it].first.second;
            const T domain_value = ldata[*it].second;

            std::set<std::pair<std::size_t, std::size_t> >::const_iterator process_data;
            for (process_data = processes_data.begin();
                 process_data != processes_data.end(); ++process_data)
            {
              const std::size_t proc = process_data->first;
              const std::size_t local_cell_entity = process_data->second;

              send_data0[proc].push_back(local_cell_entity);
              send_data0[proc].push_back(local_entity_index);
              send_data1[proc].push_back(domain_value);
            }
          }
        }
      }
    }

    // Send/receive data
    std::vector<std::vector<std::size_t> > received_data0;
    std::vector<std::vector<T> > received_data1;
    MPI::all_to_all(send_data0, received_data0);
    MPI::all_to_all(send_data1, received_data1);

    // Add received data to mesh domain
    for (std::size_t p = 0; p < num_processes; ++p)
    {
      dolfin_assert(2*received_data1[p].size() == received_data0[p].size());
      for (std::size_t i = 0; i < received_data1[p].size(); ++i)
      {
        const std::size_t local_cell_entity = received_data0[p][2*i];
        const std::size_t local_entity_index = received_data0[p][2*i + 1];
        const T value = received_data1[p][i];
        dolfin_assert(local_cell_entity < mesh.num_cells());
        markers.set_value(local_cell_entity, local_entity_index, value);
      }
    }
  }
  //---------------------------------------------------------------------------

}

#endif
