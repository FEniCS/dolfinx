// Copyright (C) 2008-2014 Niclas Jansson, Ola Skavhaug, Anders Logg
// Garth N. Wells and Chris Richardson
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
// Modified by Kent-Andre Mardal 2011
// Modified by Anders Logg 2011
// Modified by Garth N. Wells 2011-2012
// Modified by Chris Richardson 2013-2014
//

#include <algorithm>
#include <iterator>
#include <map>
#include <numeric>
#include <set>
#include <boost/multi_array.hpp>

#include <dolfin/log/log.h>
#include <dolfin/common/MPI.h>
#include <dolfin/common/Timer.h>
#include <dolfin/geometry/Point.h>
#include <dolfin/graph/ParMETIS.h>
#include <dolfin/graph/SCOTCH.h>
#include <dolfin/graph/ZoltanPartition.h>
#include <dolfin/parameter/GlobalParameters.h>

#include "DistributedMeshTools.h"
#include "Facet.h"
#include "LocalMeshData.h"
#include "Mesh.h"
#include "MeshEditor.h"
#include "MeshEntity.h"
#include "MeshEntityIterator.h"
#include "MeshFunction.h"
#include "MeshTopology.h"
#include "MeshValueCollection.h"
#include "Vertex.h"
#include "MeshPartitioning.h"

using namespace dolfin;

// Explicitly instantiate some templated functions to help the Python
// wrappers
template void MeshPartitioning::build_mesh_value_collection(const Mesh& mesh,
   const std::vector<std::pair<std::pair<std::size_t, std::size_t>, std::size_t> >&
                                                            local_value_data,
   MeshValueCollection<std::size_t>& mesh_values);

template void MeshPartitioning::build_mesh_value_collection(const Mesh& mesh,
   const std::vector<std::pair<std::pair<std::size_t, std::size_t>, int> >&
                                                            local_value_data,
   MeshValueCollection<int>& mesh_values);

template void MeshPartitioning::build_mesh_value_collection(const Mesh& mesh,
   const std::vector<std::pair<std::pair<std::size_t, std::size_t>, double> >&
                                                            local_value_data,
   MeshValueCollection<double>& mesh_values);

template void MeshPartitioning::build_mesh_value_collection(const Mesh& mesh,
   const std::vector<std::pair<std::pair<std::size_t, std::size_t>, bool> >&
                                                            local_value_data,
                                     MeshValueCollection<bool>& mesh_values);

//-----------------------------------------------------------------------------
void MeshPartitioning::build_distributed_mesh(Mesh& mesh)
{
  if (MPI::size(mesh.mpi_comm()) > 1)
  {
    // Create and distribute local mesh data
    LocalMeshData local_mesh_data(mesh);

    // Build distributed mesh
    build_distributed_mesh(mesh, local_mesh_data);
  }
}
//-----------------------------------------------------------------------------
void MeshPartitioning::build_distributed_mesh(Mesh& mesh,
                            const std::vector<std::size_t>& cell_destinations)
{
  if (MPI::size(mesh.mpi_comm()) > 1)
  {
    // Create and distribute local mesh data
    LocalMeshData local_mesh_data(mesh);

    // Attach cell destinations
    local_mesh_data.cell_partition = cell_destinations;

    // Build distributed mesh
    build_distributed_mesh(mesh, local_mesh_data);
  }
}
//-----------------------------------------------------------------------------
std::set<std::size_t> MeshPartitioning::cell_vertex_set(
    const boost::multi_array<std::size_t, 2>& cell_vertices)
{
  std::set<std::size_t> vertex_set;
  boost::multi_array<std::size_t, 2>::const_iterator vertices;
  for (vertices = cell_vertices.begin(); vertices != cell_vertices.end();
       ++vertices)
    vertex_set.insert(vertices->begin(), vertices->end());

  return vertex_set;
}
//-----------------------------------------------------------------------------
void MeshPartitioning::build_distributed_mesh(Mesh& mesh,
                                              const LocalMeshData& local_data)
{
  // Compute cell partitioning or use partitioning provided in local_data
  std::vector<std::size_t> cell_partition;
  std::map<std::size_t, dolfin::Set<unsigned int> > ghost_procs;
  if (local_data.cell_partition.empty())
    partition_cells(mesh.mpi_comm(), local_data, cell_partition, ghost_procs);
  else
  {
    cell_partition = local_data.cell_partition;
    dolfin_assert(cell_partition.size()
                  == local_data.global_cell_indices.size());
    dolfin_assert(*std::max_element(cell_partition.begin(), cell_partition.end())
                  < MPI::size(mesh.mpi_comm()));
  }

  // Build mesh from local mesh data and provided cell partition
  build(mesh, local_data, cell_partition, ghost_procs);

  // Create MeshDomains from local_data
  build_mesh_domains(mesh, local_data);

  // Initialise number of globally connected cells to each facet. This is
  // necessary to distinguish between facets on an exterior boundary and
  // facets on a partition boundary (see
  // https://bugs.launchpad.net/dolfin/+bug/733834).

  // FIXME: make it work again
  //  DistributedMeshTools::init_facet_cell_connections_by_ghost(mesh);
}
//-----------------------------------------------------------------------------
void MeshPartitioning::partition_cells(const MPI_Comm& mpi_comm,
      const LocalMeshData& mesh_data,
      std::vector<std::size_t>& cell_partition,
      std::map<std::size_t, dolfin::Set<unsigned int> >& ghost_procs)
{

  // Compute cell partition using partitioner from parameter system
  const std::string partitioner = parameters["mesh_partitioner"];
  if (partitioner == "SCOTCH")
    SCOTCH::compute_partition(mpi_comm, cell_partition, ghost_procs, mesh_data);
  else if (partitioner == "ParMETIS")
    ParMETIS::compute_partition(mpi_comm, cell_partition, ghost_procs, mesh_data);
  else if (partitioner == "Zoltan_RCB")
  {
    ZoltanPartition::compute_partition_rcb(mpi_comm, cell_partition,
                                           mesh_data);
  }
  else if (partitioner == "Zoltan_PHG")
  {
    ZoltanPartition::compute_partition_phg(mpi_comm, cell_partition,
                                           mesh_data);
  }
  else
  {
    dolfin_error("MeshPartitioning.cpp",
                 "compute cell partition",
                 "Mesh partitioner '%s' is unknown.", partitioner.c_str());
  }
}
//-----------------------------------------------------------------------------
void MeshPartitioning::build(Mesh& mesh, const LocalMeshData& mesh_data,
     const std::vector<std::size_t>& cell_partition,
     const std::map<std::size_t, dolfin::Set<unsigned int> >& ghost_procs)
{

  // Distribute cells
  Timer timer("PARALLEL 2: Distribute mesh (cells and vertices)");

  // Structure to hold received data about local mesh
  LocalMeshData new_mesh_data(mesh.mpi_comm());

  // Copy over some basic information
  new_mesh_data.tdim = mesh_data.tdim;
  new_mesh_data.gdim = mesh_data.gdim;
  new_mesh_data.num_global_cells = mesh_data.num_global_cells;
  new_mesh_data.num_global_vertices = mesh_data.num_global_vertices;  

  // Keep tabs on ghost cell ownership
  std::map<unsigned int, std::set<unsigned int> > shared_cells;
  // Send cells to processes that need them
  distribute_ghost_cells(mesh.mpi_comm(), mesh_data,
                         cell_partition, ghost_procs, 
                         shared_cells, new_mesh_data);

  // Create map of shared vertices before distribution.
  // Use global indexing, since at this point, 
  // the local vertex indexing is not yet assigned.
  std::map<std::size_t, std::set<unsigned int> > shared_vertices_global;
  ghost_build_shared_vertices(mesh.mpi_comm(), new_mesh_data,
                              shared_cells,
                              shared_vertices_global);

  distribute_cell_layer(mesh.mpi_comm(), 
                        shared_vertices_global,
                        shared_cells,
                        new_mesh_data);

  // Distribute vertices into new_mesh_data structure
  const std::set<std::size_t> vertex_set 
    = cell_vertex_set(new_mesh_data.cell_vertices);
  std::map<std::size_t, std::size_t> vertex_global_to_local;
  distribute_vertices(mesh.mpi_comm(), mesh_data, vertex_set, 
                      new_mesh_data.vertex_indices,
                      vertex_global_to_local, 
                      new_mesh_data.vertex_coordinates);

  timer.stop();

  // Build mesh from new_mesh_data
  build_mesh(mesh, vertex_global_to_local, new_mesh_data);

  // Convert shared_vertices to local indexing
  std::map<unsigned int, std::set<unsigned int> >& 
    shared_vertices_local = mesh.topology().shared_entities(0);
  for (auto map_it = shared_vertices_global.begin(); 
       map_it != shared_vertices_global.end(); ++map_it)
  {
    const std::size_t g_index = map_it->first;
    dolfin_assert(vertex_global_to_local.find(g_index) 
                  != vertex_global_to_local.end());
    const unsigned int l_index = vertex_global_to_local[g_index];
    shared_vertices_local.insert
      (std::make_pair(l_index, map_it->second));
  }

  // FIXME: this may later become a Mesh member variable  
  mesh.data().create_array("ghost_owner", mesh_data.tdim);
  // Copy array over
  std::vector<std::size_t>& ghost_cell_owner = 
    mesh.data().array("ghost_owner", mesh_data.tdim);
  ghost_cell_owner.insert(ghost_cell_owner.begin(),
                          new_mesh_data.cell_partition.begin(),
                          new_mesh_data.cell_partition.end());

  // Assign map of shared cells
  mesh.topology().shared_entities(mesh_data.tdim) = shared_cells;

}
//-----------------------------------------------------------------------------
void MeshPartitioning::distribute_cell_layer(MPI_Comm mpi_comm,
  const std::map<std::size_t, std::set<unsigned int> >& shared_vertices_global,
  const std::map<unsigned int, std::set<unsigned int> > shared_cells,
  LocalMeshData& new_mesh_data)
{
  const unsigned int mpi_size = MPI::size(mpi_comm);
  const unsigned int mpi_rank = MPI::rank(mpi_comm);

  // Make a simple list of shared vertices (global indexing)
  std::vector<std::size_t> sh_vert_list;
  for (auto v = shared_vertices_global.begin();
       v != shared_vertices_global.end(); ++v)
    sh_vert_list.push_back(v->first);

  // Find all local cells (local indexing) attached to vertices
  std::map<std::size_t, dolfin::Set<std::size_t> >
    sh_vert_to_cell = cell_attachment(sh_vert_list, new_mesh_data);

  // Map from process to cells it should have (some may already be there)
  std::map<unsigned int, dolfin::Set<std::size_t> > proc_cells;

  for (auto v = sh_vert_to_cell.begin(); 
       v != sh_vert_to_cell.end(); ++v)
  {
    const std::size_t v_index = v->first;
    auto map_it = shared_vertices_global.find(v_index);
    dolfin_assert(map_it != shared_vertices_global.end());
    // Set of processes attached to vertex
    const std::set<unsigned int>& v_procs = map_it->second;
    
    // Limit set of cells to those with local ownership
    dolfin::Set<std::size_t> v_cells;
    for (auto c = v->second.begin(); c != v->second.end(); ++c)
    {
      if (new_mesh_data.cell_partition[*c] == mpi_rank)
        v_cells.insert(*c);
    }

    for (auto p = v_procs.begin(); p != v_procs.end(); ++p)
    {
      auto pc_it = proc_cells.find(*p);
      if (pc_it == proc_cells.end())
        proc_cells.insert(std::make_pair(*p, v_cells));
      else
        pc_it->second.insert(v_cells.begin(), v_cells.end());
    }
  }

  // Send from cell owner to where cells are needed

  std::vector<std::vector<std::size_t> > send_cells(mpi_size);
  std::vector<std::vector<std::size_t> > recv_cells(mpi_size);

  for (auto p = proc_cells.begin(); p != proc_cells.end(); ++p)
  {
    std::vector<std::size_t>& send_dest = send_cells[p->first];
    for (auto c = p->second.begin(); c != p->second.end(); ++c)
    {
      // Get list of processes this cell is already shared with,
      // if any
      auto cell_it = shared_cells.find(*c);
      std::set<unsigned int> proc_set;
      if (cell_it == shared_cells.end())
      {
        proc_set.insert(mpi_rank);
        // FIXME: cell is now shared 
        // add send_dest to shared_cells
      }
      else
      {
        proc_set = cell_it->second;
        // FIXME: check if send_dest is already in proc_set
        // If so, no need to send
      }

      // Send cell global index
      send_dest.push_back(new_mesh_data.global_cell_indices[*c]);

      // Count of ghosts, followed by ghost processes
      send_dest.push_back(proc_set.size());
      for (auto p = proc_set.begin(); p != proc_set.end(); ++p)
        send_dest.push_back(*p);
      
      // Global vertex indices
      send_dest.insert(send_dest.end(), 
                       new_mesh_data.cell_vertices[*c].begin(),
                       new_mesh_data.cell_vertices[*c].end());
    }
  }

  MPI::all_to_all(mpi_comm, send_cells, recv_cells);

  const unsigned int num_cells
    = new_mesh_data.cell_vertices.shape()[0];
  const unsigned int num_cell_vertices 
    = new_mesh_data.cell_vertices.shape()[1];
  
  std::map<std::size_t, std::vector<std::size_t> > new_cells;
  for (auto p = recv_cells.begin(); p != recv_cells.end(); ++p)
  {
    unsigned int mpi_sender = p - recv_cells.begin();
    
    auto q = p->begin();
    while (q != p->end())
    {
      const std::size_t g_index = *q++;
      const unsigned int nghosts = *q;
      const std::vector<std::size_t> cell_data(q, 
                        q + nghosts + num_cell_vertices + 1);
      
      auto map_it = new_cells.find(g_index);
      dolfin_assert(map_it == new_cells.end());
      
      // Seek global index
      auto gp = std::find(new_mesh_data.global_cell_indices.begin(),
                          new_mesh_data.global_cell_indices.end(),
                          g_index);
      if (gp == new_mesh_data.global_cell_indices.end())
      {        
        // Cell does not exist locally
        std::pair<std::map<std::size_t, std::vector<std::size_t> >::iterator,
                  bool> new_it 
          = new_cells.insert(std::make_pair(g_index, cell_data));
        dolfin_assert(new_it.second);
        // Attach sender information
        new_it.first->second.push_back(mpi_sender);
      }
      // FIXME - if not new cell, add mpi_sender to sharing information
      q += nghosts + num_cell_vertices + 1;
    }
  }

  // Add extra data to new_mesh_data
  new_mesh_data.cell_vertices.resize
    (boost::extents[num_cells + new_cells.size()][num_cell_vertices]);
  new_mesh_data.cell_partition.resize
    (num_cells + new_cells.size());
  new_mesh_data.global_cell_indices.resize
    (num_cells + new_cells.size());
  
  unsigned int c = num_cells;
  for (auto p = new_cells.begin(); p != new_cells.end(); ++p)
  {
    new_mesh_data.global_cell_indices[c] = p->first;
    const std::vector<std::size_t>& cell_data = p->second;
    new_mesh_data.cell_partition[c] = cell_data.back();
    const unsigned int nghosts = cell_data[0];
    for (unsigned int j = 0; j != num_cell_vertices; ++j)
      new_mesh_data.cell_vertices[c][j]
        = cell_data[nghosts + 1 + j];
    ++c;
  }
  

}
//-----------------------------------------------------------------------------
void MeshPartitioning::ghost_build_shared_vertices(MPI_Comm mpi_comm,
  const LocalMeshData& new_mesh_data,
  const std::map<unsigned int, std::set<unsigned int> > shared_cells,
  std::map<std::size_t, std::set<unsigned int> >& shared_vertices_global)

{
  const boost::multi_array<std::size_t, 2>& cell_vertices 
    = new_mesh_data.cell_vertices;
  
  std::map<std::size_t, dolfin::Set<unsigned int> > global_vertex_to_procs;

  // Iterate over all vertices of each cell in the shared_cells map
  for (auto map_it = shared_cells.begin(); 
       map_it != shared_cells.end(); ++map_it)
  {
    const std::size_t cell_i = map_it->first;
    const unsigned int cell_owner = new_mesh_data.cell_partition[cell_i];
    
    for (auto q = cell_vertices[cell_i].begin(); 
         q != cell_vertices[cell_i].end(); ++q)
    {
      auto vtx_it = global_vertex_to_procs.find(*q);
      if (vtx_it == global_vertex_to_procs.end())
      {
        dolfin::Set<unsigned int> proc_set;
        proc_set.insert(cell_owner);
        global_vertex_to_procs.insert(std::make_pair(*q, proc_set));
      }
      else
        vtx_it->second.insert(cell_owner);
    }
  }

  // Erase any entries which are 'local'
  auto map_it = global_vertex_to_procs.begin();
  while (map_it != global_vertex_to_procs.end())
  {
    std::cout << map_it->second.size() << "\n";
    if (map_it->second.size() == 1)
      global_vertex_to_procs.erase(map_it++);
    else
      map_it++;
  }

  // N.B. there are some special cases where care must be taken,
  // particularly where vertices are shared between multiple (>2)
  // processes. The safest approach (?) is to send all vertex information
  // to a sorting process, which then redistributes to the sharing
  // processes.
  
  const unsigned int mpi_size = MPI::size(mpi_comm);

  std::vector<std::vector<std::size_t> > send_vertex(mpi_size);
  std::vector<std::vector<std::size_t> > recv_vertex(mpi_size);
  
  for (auto map_it = global_vertex_to_procs.begin();
       map_it != global_vertex_to_procs.end(); ++map_it)
  {
    const std::size_t vertex_i = map_it->first;
    const unsigned int dest = MPI::index_owner(mpi_comm, vertex_i, 
                                    new_mesh_data.num_global_vertices);
    std::vector<std::size_t>& send_dest = send_vertex[dest];
    
    send_dest.push_back(vertex_i);
    send_dest.push_back(map_it->second.size());
    send_dest.insert(send_dest.end(), map_it->second.begin(),
                     map_it->second.end());    
  }

  // Send vertex map to sorting process (vertex index owner)
  MPI::all_to_all(mpi_comm, send_vertex, recv_vertex);

  // Clear send vector to send vertices back out
  send_vertex = std::vector<std::vector<std::size_t> >(mpi_size);
  
  // Send vertices to all processes which share them
  // with list of sharing processes
  for (auto p = recv_vertex.begin(); p != recv_vertex.end(); ++p)
  {
    auto q = p->begin();
    while (q != p->end())
    {
      const std::size_t v_index = *q++;
      const unsigned int nprocs = *q++;
      for (auto proc_it = q; proc_it != q + nprocs; ++proc_it)
      {
        std::vector<std::size_t>& send_dest = send_vertex[*proc_it];
        send_dest.push_back(v_index);
        send_dest.push_back(nprocs);
        send_dest.insert(send_dest.end(), q, q + nprocs);
      }
      q += nprocs;
    }
  }
  
  MPI::all_to_all(mpi_comm, send_vertex, recv_vertex);
  
  // Insert received vertex sharing information into map
  for (auto p = recv_vertex.begin(); p != recv_vertex.end(); ++p)
  {
    auto q = p->begin();
    while (q != p->end())
    {
      const std::size_t v_index = *q++;
      const unsigned int nprocs = *q++;
      const std::set<unsigned int> proc_set(q, q + nprocs);
      auto map_it = shared_vertices_global.find(v_index);
      if (map_it == shared_vertices_global.end())
        shared_vertices_global.insert(std::make_pair(v_index, proc_set));
      else
        map_it->second.insert(proc_set.begin(), proc_set.end());      
      q += nprocs;
    }
  }

}
//-----------------------------------------------------------------------------
std::map<std::size_t, dolfin::Set<std::size_t> >
MeshPartitioning::cell_attachment(const std::vector<std::size_t> vertex_list,
                                  const LocalMeshData& mesh_data)
{
  const boost::multi_array<std::size_t, 2>& cell_vertices
    = mesh_data.cell_vertices;

  std::map<std::size_t, dolfin::Set<std::size_t> > attachment_map;

  // Initialise empty map
  for (auto v = vertex_list.begin(); v != vertex_list.end(); ++v)
    attachment_map.insert(std::make_pair(*v, dolfin::Set<std::size_t>()));

  // Go through all cell vertices, looking for any in list
  for (unsigned int i = 0; i != cell_vertices.size(); ++i)
  {
    for (auto v = cell_vertices[i].begin(); 
         v != cell_vertices[i].end(); ++v)
    {
      auto map_it = attachment_map.find(*v);
      if (map_it != attachment_map.end())
        map_it->second.insert(i);
    }
  }
  return attachment_map;
}
//-----------------------------------------------------------------------------
void MeshPartitioning::distribute_ghost_cells(const MPI_Comm mpi_comm,
      const LocalMeshData& mesh_data,
      const std::vector<std::size_t>& cell_partition,
      const std::map<std::size_t, dolfin::Set<unsigned int> >& ghost_procs,
      std::map<unsigned int, std::set<unsigned int> >& shared_cells,
      LocalMeshData& new_mesh_data)
{
  // This function takes the partition computed by the partitioner
  // stored in cell_partition/ghost_procs 
  // Some cells go to multiple destinations.
  // Each cell is transmitted to its final destination(s) including
  // its global index, and the cell owner (for ghost cells this
  // will be different from the destination)

  const std::size_t num_processes = MPI::size(mpi_comm);
  const std::size_t mpi_rank = MPI::rank(mpi_comm);

  // Get dimensions of local mesh_data
  const std::size_t num_local_cells = mesh_data.cell_vertices.size();
  dolfin_assert(mesh_data.global_cell_indices.size() == num_local_cells);
  const std::size_t num_cell_vertices = mesh_data.num_vertices_per_cell;
  if (!mesh_data.cell_vertices.empty())
  {
    if (mesh_data.cell_vertices[0].size() != num_cell_vertices)
    {
      dolfin_error("MeshPartitioning.cpp",
                   "distribute cells",
                   "Mismatch in number of cell vertices (%d != %d) on process %d",
                   mesh_data.cell_vertices[0].size(), num_cell_vertices,
                   mpi_rank);
    }
  }

  // Send all cells to their destinations including their global indices. 
  // First element of vector is cell count.
  std::vector<std::vector<std::size_t> > send_cell_vertices(num_processes,
                               std::vector<std::size_t>(1, 0) );
  
  for (unsigned int i = 0; i != cell_partition.size(); ++i)
  {
    // If cell is in ghost_procs map, use that to determine
    // destinations, otherwise just use the cell_partition
    // vector

    auto map_it = ghost_procs.find(i);
    if (map_it != ghost_procs.end()) 
    {
      const dolfin::Set<unsigned int>& destinations
        = map_it->second;
      
      for (auto dest = destinations.begin(); 
           dest != destinations.end(); ++dest)
      {
        // Count of ghost cells, followed by ghost processes
        send_cell_vertices[*dest].push_back(destinations.size());
        for (std::size_t j = 0; j < destinations.size(); j++)
          send_cell_vertices[*dest].push_back(destinations[j]);

        // Global cell index
        send_cell_vertices[*dest].push_back
          (mesh_data.global_cell_indices[i]); 
        // Global vertex indices
        for (std::size_t j = 0; j < num_cell_vertices; j++)
          send_cell_vertices[*dest].push_back
            (mesh_data.cell_vertices[i][j]);
        send_cell_vertices[*dest][0]++;
      }
    }
    else
    {
      // Single destination (unghosted cell)
      const unsigned int dest = cell_partition[i];
      send_cell_vertices[dest].push_back(0);
      // Global cell index
      send_cell_vertices[dest].push_back
        (mesh_data.global_cell_indices[i]); 
      // Global vertex indices
      for (std::size_t j = 0; j < num_cell_vertices; j++)
        send_cell_vertices[dest].push_back
          (mesh_data.cell_vertices[i][j]);
      send_cell_vertices[dest][0]++;
    }
  }

  // Distribute cell-vertex connectivity
  std::vector<std::vector<std::size_t> > received_cell_vertices(num_processes);
  MPI::all_to_all(mpi_comm, send_cell_vertices, received_cell_vertices);

  // Count number of received cells
  std::size_t num_new_local_cells = 0;
  for (std::size_t p = 0; p < num_processes; ++p)
    num_new_local_cells += received_cell_vertices[p][0]; 

  // Put received mesh data into new_mesh_data structure
  new_mesh_data.cell_vertices.resize(boost::extents[num_new_local_cells]
                                     [num_cell_vertices]);
  new_mesh_data.global_cell_indices.resize(num_new_local_cells);
  new_mesh_data.cell_partition.resize(num_new_local_cells);

  // Loop over new cells
  std::size_t c = 0;
  for (std::size_t p = 0; p < num_processes; ++p)
  {
    std::vector<std::size_t>& received_data = received_cell_vertices[p];
    for (auto it = received_data.begin() + 1;
         it != received_data.end();
         it += (*it + num_cell_vertices + 2))
    {
      auto tmp_it = it;
      const unsigned int num_ghosts = *tmp_it++;
      
      if (num_ghosts == 0)
        new_mesh_data.cell_partition[c] = mpi_rank;
      else
      {
        new_mesh_data.cell_partition[c] = *tmp_it;
        std::set<unsigned int> proc_set(tmp_it, tmp_it + num_ghosts);
        // Remove self from set of sharing processes
        proc_set.erase(mpi_rank);
        shared_cells.insert(std::make_pair(c, proc_set));
      }

      tmp_it += num_ghosts;
      new_mesh_data.global_cell_indices[c] = *tmp_it++;

      for (std::size_t j = 0; j < num_cell_vertices; ++j)
        new_mesh_data.cell_vertices[c][j] = *tmp_it++;
      ++c;
    }
  }

}
//-----------------------------------------------------------------------------
void MeshPartitioning::distribute_vertices(const MPI_Comm mpi_comm,
                    const LocalMeshData& mesh_data,
                    const std::set<std::size_t>& needed_vertex_indices,
                    std::vector<std::size_t>& vertex_indices,
                    std::map<std::size_t, std::size_t>& vertex_global_to_local,
                    boost::multi_array<double, 2>& vertex_coordinates)
{
  // This function distributes all vertices (coordinates and
  // local-to-global mapping) according to the cells that are stored on
  // each process. This happens in several stages: First each process
  // figures out which vertices it needs (by looking at its cells)
  // and where those vertices are located. That information is then
  // distributed so that each process learns where it needs to send
  // its vertices.

  // Get number of processes
  const std::size_t num_processes = MPI::size(mpi_comm);

  // Get geometric dimension
  const std::size_t gdim = mesh_data.gdim;

  // Compute where (process number) the vertices we need are located
  std::vector<std::vector<std::size_t> > send_vertex_indices(num_processes);
  //  std::vector<std::vector<std::size_t> > vertex_location(num_processes);
  std::set<std::size_t>::const_iterator required_vertex;
  for (required_vertex = needed_vertex_indices.begin();
       required_vertex != needed_vertex_indices.end(); ++required_vertex)
  {
    // Get process that has required vertex
    const std::size_t location = MPI::index_owner(mpi_comm, *required_vertex,
                                                mesh_data.num_global_vertices);
    send_vertex_indices[location].push_back(*required_vertex);
    //    vertex_location[location].push_back(*required_vertex);
  }

  const std::vector<std::vector<std::size_t> >& vertex_location = send_vertex_indices;

  // Send required vertices to other processes, and receive back vertices
  // required by other processes.
  std::vector<std::vector<std::size_t> > received_vertex_indices;
  MPI::all_to_all(mpi_comm, send_vertex_indices, received_vertex_indices);

  // Distribute vertex coordinates
  std::vector<std::vector<double> > send_vertex_coordinates(num_processes);
  const std::pair<std::size_t, std::size_t> local_vertex_range
    = MPI::local_range(mpi_comm, mesh_data.num_global_vertices);
  for (std::size_t p = 0; p < num_processes; ++p)
  {
    send_vertex_coordinates[p].reserve(received_vertex_indices[p].size()*gdim);
    for (std::size_t i = 0; i < received_vertex_indices[p].size(); ++i)
    {
      dolfin_assert(received_vertex_indices[p][i] >= local_vertex_range.first
                 && received_vertex_indices[p][i] < local_vertex_range.second);
      const std::size_t location
        = received_vertex_indices[p][i] - local_vertex_range.first;
      for (std::size_t j = 0; j < gdim; ++j)
        send_vertex_coordinates[p].push_back(mesh_data.vertex_coordinates[location][j]);
    }
  }
  std::vector<std::vector<double> > received_vertex_coordinates;
  MPI::all_to_all(mpi_comm, send_vertex_coordinates,
                  received_vertex_coordinates);

  // Count number of new local vertices
  std::size_t num_local_vertices = vertex_indices.size();
  std::size_t v = num_local_vertices;
  dolfin_assert(num_local_vertices == vertex_coordinates.size());
  for (std::size_t p = 0; p < num_processes; ++p)
    num_local_vertices += received_vertex_coordinates[p].size()/gdim;

  // Store coordinates and construct global to local mapping
  vertex_coordinates.resize(boost::extents[num_local_vertices][gdim]);
  vertex_indices.resize(num_local_vertices);

  for (std::size_t p = 0; p < num_processes; ++p)
  {
    for (std::size_t i = 0;
         i < received_vertex_coordinates[p].size()/gdim; ++i)
    {
      for (std::size_t j = 0; j < gdim; ++j)
        vertex_coordinates[v][j]
          = received_vertex_coordinates[p][i*gdim + j];

      const std::size_t global_vertex_index
        = vertex_location[p][i];
      vertex_global_to_local[global_vertex_index] = v;
      vertex_indices[v] = global_vertex_index;

      ++v;
    }
  }
}
//-----------------------------------------------------------------------------
void MeshPartitioning::build_mesh(Mesh& mesh,
  const std::map<std::size_t, std::size_t>& vertex_global_to_local,
  const LocalMeshData& new_mesh_data)
{
  Timer timer("PARALLEL 3: Build mesh (from local mesh data)");

  const std::vector<std::size_t>& global_cell_indices
    = new_mesh_data.global_cell_indices;
  const boost::multi_array<std::size_t, 2>& cell_global_vertices
    = new_mesh_data.cell_vertices;
  const std::vector<std::size_t>& vertex_indices
    = new_mesh_data.vertex_indices;
  const boost::multi_array<double, 2>& vertex_coordinates
    = new_mesh_data.vertex_coordinates;

  const unsigned int gdim = new_mesh_data.gdim;
  const unsigned int tdim = new_mesh_data.tdim;

  // Open mesh for editing
  mesh.clear();
  MeshEditor editor;
  editor.open(mesh, tdim, gdim);

  // Add vertices
  editor.init_vertices_global(vertex_coordinates.size(), 
                              new_mesh_data.num_global_vertices);
  Point point(gdim);
  dolfin_assert(vertex_indices.size() == vertex_coordinates.size());
  for (std::size_t i = 0; i < vertex_coordinates.size(); ++i)
  {
    for (std::size_t j = 0; j < gdim; ++j)
      point[j] = vertex_coordinates[i][j];
    editor.add_vertex_global(i, vertex_indices[i], point);
  }

  // Add cells
  editor.init_cells_global(cell_global_vertices.size(), 
                           new_mesh_data.num_global_cells);
  const std::size_t num_cell_vertices = tdim + 1;
  std::vector<std::size_t> cell(num_cell_vertices);
  for (std::size_t i = 0; i < cell_global_vertices.size(); ++i)
  {
    for (std::size_t j = 0; j < num_cell_vertices; ++j)
    {
      // Get local cell vertex
      std::map<std::size_t, std::size_t>::const_iterator iter
          = vertex_global_to_local.find(cell_global_vertices[i][j]);
      dolfin_assert(iter != vertex_global_to_local.end());
      cell[j] = iter->second;
    }
    editor.add_cell(i, global_cell_indices[i], cell);
  }

  // Close mesh: Note that this must be done after creating the global
  // vertex map or otherwise the ordering in mesh.close() will be wrong
  // (based on local numbers).
  editor.close();

  // Set global number of cells and vertices
  // mesh.topology().init_global(0, num_global_vertices);
  //   mesh.topology().init_global(tdim,  num_global_cells);
}
//-----------------------------------------------------------------------------
void MeshPartitioning::build_mesh_domains(Mesh& mesh,
                                          const LocalMeshData& local_data)
{
  // Local domain data
  const std::map<std::size_t,  std::vector<
    std::pair<std::pair<std::size_t, std::size_t>, std::size_t> > >&
    domain_data = local_data.domain_data;

  if (domain_data.empty())
    return;

  // Initialise mesh domains
  const std::size_t D = mesh.topology().dim();
  mesh.domains().init(D);

  std::map<std::size_t, std::vector<
    std::pair<std::pair<std::size_t, std::size_t>,
              std::size_t> > >::const_iterator dim_data;
  for (dim_data = domain_data.begin(); dim_data != domain_data.end();
       ++dim_data)
  {
    // Get mesh value collection used for marking
    const std::size_t dim = dim_data->first;

    // Initialise mesh
    mesh.init(dim);

    // Create empty MeshValueCollection
    MeshValueCollection<std::size_t> mvc(mesh, dim);

    // Get domain data
    const std::vector<std::pair<std::pair<std::size_t, std::size_t>,
                                std::size_t> >& local_value_data
                                = dim_data->second;

    // Build mesh value collection
    build_mesh_value_collection(mesh, local_value_data, mvc);

    // Get data from mesh value collection
    const std::map<std::pair<std::size_t, std::size_t>, std::size_t>& values
      = mvc.values();

    // Get map from mesh domains
    std::map<std::size_t, std::size_t>& markers = mesh.domains().markers(dim);

    std::map<std::pair<std::size_t, std::size_t>,
             std::size_t>::const_iterator it;
    for (it = values.begin(); it != values.end(); ++it)
    {
      const std::size_t cell_index = it->first.first;
      const std::size_t local_entity_index = it->first.second;

      const Cell cell(mesh, cell_index);
      const MeshEntity e(mesh, dim, cell.entities(dim)[local_entity_index]);
      markers[e.index()] = it->second;
    }
  }
}
//-----------------------------------------------------------------------------
