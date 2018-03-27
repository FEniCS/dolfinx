// Copyright (C) 2008-2014 Niclas Jansson, Ola Skavhaug, Anders Logg,
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "MeshPartitioning.h"
#include "CellType.h"
#include "DistributedMeshTools.h"
#include "Facet.h"
#include "LocalMeshData.h"
#include "Mesh.h"
#include "MeshEntity.h"
#include "MeshFunction.h"
#include "MeshTopology.h"
#include "MeshValueCollection.h"
#include "Vertex.h"
#include <algorithm>
#include <boost/multi_array.hpp>
#include <cstdint>
#include <dolfin/common/MPI.h>
#include <dolfin/common/Timer.h>
#include <dolfin/geometry/Point.h>
#include <dolfin/graph/GraphBuilder.h>
#include <dolfin/graph/ParMETIS.h>
#include <dolfin/graph/SCOTCH.h>
#include <dolfin/log/log.h>
#include <dolfin/parameter/GlobalParameters.h>
#include <iterator>
#include <map>
#include <memory>
#include <numeric>
#include <set>

using namespace dolfin;
using namespace dolfin::mesh;

//-----------------------------------------------------------------------------
mesh::Mesh MeshPartitioning::build_distributed_mesh(const Mesh& mesh)
{
  // Create and distribute local mesh data
  LocalMeshData local_mesh_data(mesh);

  // Build distributed mesh
  return build_distributed_mesh(local_mesh_data,
                                parameter::parameters["ghost_mode"]);
}
//-----------------------------------------------------------------------------
mesh::Mesh MeshPartitioning::build_distributed_mesh(
    const Mesh& mesh, const std::vector<int>& cell_destinations,
    const std::string ghost_mode)
{
  // Create and distribute local mesh data
  LocalMeshData local_mesh_data(mesh);

  // Attach cell destinations
  local_mesh_data.topology.cell_partition = cell_destinations;

  // Build distributed mesh
  return build_distributed_mesh(local_mesh_data, ghost_mode);
}
//-----------------------------------------------------------------------------
mesh::Mesh
MeshPartitioning::build_distributed_mesh(const LocalMeshData& local_data,
                                         const std::string ghost_mode)
{
  log::log(PROGRESS, "Building distributed mesh");

  common::Timer timer("Build distributed mesh from local mesh data");

  // MPI communicator
  MPI_Comm comm = local_data.mpi_comm();

  // Get mesh partitioner
  const std::string partitioner = parameter::parameters["mesh_partitioner"];

  // Compute cell partitioning or use partitioning provided in local_data
  std::vector<int> cell_partition;
  std::map<std::int64_t, std::vector<int>> ghost_procs;
  if (local_data.topology.cell_partition.empty())
    partition_cells(comm, local_data, partitioner, cell_partition, ghost_procs);
  else
  {
    // Copy cell partition
    cell_partition = local_data.topology.cell_partition;
    dolfin_assert(cell_partition.size()
                  == local_data.topology.global_cell_indices.size());
    dolfin_assert(
        *std::max_element(cell_partition.begin(), cell_partition.end())
        < (int)MPI::size(comm));
  }

  // Check that we have some ghost information.
  int all_ghosts = MPI::sum(comm, ghost_procs.size());
  if (all_ghosts == 0 && ghost_mode != "none")
  {
    // FIXME: need to generate ghost cell information here by doing a
    // facet-matching operation "GraphBuilder" style
    log::dolfin_error("MeshPartitioning.cpp", "build ghost mesh",
                      "Ghost cell information not available");
  }

  // Build mesh from local mesh data and provided cell partition
  mesh::Mesh mesh
      = build(comm, local_data, cell_partition, ghost_procs, ghost_mode);

  // Store used ghost mode
  // NOTE: This is the only place in DOLFIN which eventually sets
  //       mesh._ghost_mode != "none"
  mesh._ghost_mode = ghost_mode;
  mesh._ordered = local_data.topology.ordered;

  // Initialise number of globally connected cells to each facet. This
  // is necessary to distinguish between facets on an exterior
  // boundary and facets on a partition boundary (see
  // https://bugs.launchpad.net/dolfin/+bug/733834).

  DistributedMeshTools::init_facet_cell_connections(mesh);

  return mesh;
}
//-----------------------------------------------------------------------------
void MeshPartitioning::partition_cells(
    const MPI_Comm& mpi_comm, const LocalMeshData& mesh_data,
    const std::string partitioner, std::vector<int>& cell_partition,
    std::map<std::int64_t, std::vector<int>>& ghost_procs)
{
  log::log(PROGRESS, "Compute partition of cells across processes");

  // Clear data
  cell_partition.clear();
  ghost_procs.clear();

  std::unique_ptr<mesh::CellType> cell_type(
      mesh::CellType::create(mesh_data.topology.cell_type));
  dolfin_assert(cell_type);

  // Compute cell partition using partitioner from parameter system
  if (partitioner == "SCOTCH")
  {
    dolfin::graph::SCOTCH::compute_partition(
        mpi_comm, cell_partition, ghost_procs, mesh_data.topology.cell_vertices,
        mesh_data.topology.cell_weight, mesh_data.geometry.num_global_vertices,
        mesh_data.topology.num_global_cells, *cell_type);
  }
  else if (partitioner == "ParMETIS")
  {
    dolfin::graph::ParMETIS::compute_partition(
        mpi_comm, cell_partition, ghost_procs, mesh_data.topology.cell_vertices,
        mesh_data.geometry.num_global_vertices, *cell_type);
  }
  else
  {
    log::dolfin_error("MeshPartitioning.cpp", "compute cell partition",
                      "Mesh partitioner '%s' is unknown.", partitioner.c_str());
  }
}
//-----------------------------------------------------------------------------
mesh::Mesh MeshPartitioning::build(
    const MPI_Comm& comm, const LocalMeshData& mesh_data,
    const std::vector<int>& cell_partition,
    const std::map<std::int64_t, std::vector<int>>& ghost_procs,
    const std::string ghost_mode)
{
  // Distribute cells
  log::log(PROGRESS, "Distribute mesh (cell and vertices)");

  common::Timer timer("Distribute mesh (cells and vertices)");

  // Topological dimension
  const int tdim = mesh_data.topology.dim;

  const std::int64_t num_global_vertices
      = mesh_data.geometry.num_global_vertices;
  const std::int32_t num_cell_vertices
      = mesh_data.topology.num_vertices_per_cell;

  // FIXME: explain structure of shared_cells
  // Keep tabs on ghost cell ownership (map from local cell index to
  // sharing processes)

  // Send cells to processes that need them. Returns
  // 0. Number of regular cells on this process
  // 1. Map from local cell index to to sharing process for ghosted cells
  boost::multi_array<std::int64_t, 2> new_cell_vertices;
  std::vector<std::int64_t> new_global_cell_indices;
  std::vector<int> new_cell_partition;
  std::map<std::int32_t, std::set<std::uint32_t>> shared_cells;

  // Send cells to owning process accoring to cell_partition, and
  // receive cells that belong to this process. Also compute auxiliary
  // data related to sharing,
  const std::int32_t num_regular_cells = distribute_cells(
      comm, mesh_data, cell_partition, ghost_procs, new_cell_vertices,
      new_global_cell_indices, new_cell_partition, shared_cells);

  if (ghost_mode == "shared_vertex")
  {
    // Send/receive additional cells defined by connectivity to the shared
    // vertices.
    distribute_cell_layer(comm, num_regular_cells, num_global_vertices,
                          shared_cells, new_cell_vertices,
                          new_global_cell_indices, new_cell_partition);
  }
  else if (ghost_mode == "none")
  {
    // Resize to remove all ghost cells
    new_cell_partition.resize(num_regular_cells);
    new_global_cell_indices.resize(num_regular_cells);
    new_cell_vertices.resize(
        boost::extents[num_regular_cells][num_cell_vertices]);
    shared_cells.clear();
  }

#ifdef HAS_SCOTCH
  if (parameter::parameters["reorder_cells_gps"])
  {
    // Create CellType objects based on current cell type
    std::unique_ptr<mesh::CellType> cell_type(
        mesh::CellType::create(mesh_data.topology.cell_type));
    dolfin_assert(cell_type);

    // Allocate objects to hold re-ordering
    std::map<std::int32_t, std::set<std::uint32_t>> reordered_shared_cells;
    boost::multi_array<std::int64_t, 2> reordered_cell_vertices;
    std::vector<std::int64_t> reordered_global_cell_indices;

    // Re-order cells
    reorder_cells_gps(comm, num_regular_cells, *cell_type, shared_cells,
                      new_cell_vertices, new_global_cell_indices,
                      reordered_shared_cells, reordered_cell_vertices,
                      reordered_global_cell_indices);

    // Update to paramre-ordered indices
    std::swap(shared_cells, reordered_shared_cells);
    std::swap(new_cell_vertices, reordered_cell_vertices);
    std::swap(new_global_cell_indices, reordered_global_cell_indices);
  }
#endif

  // Generate mapping from global to local indexing for vertices also
  // calculating which vertices are 'ghost' and putting them at the
  // end of the local range
  std::vector<std::int64_t> vertex_indices;
  EigenRowArrayXXi32 local_cell_vertices(new_cell_vertices.shape()[0],
                                         new_cell_vertices.shape()[1]);
  const std::int32_t num_regular_vertices
      = compute_vertex_mapping(comm, num_regular_cells, new_cell_vertices,
                               vertex_indices, local_cell_vertices);

  // Send vertices to processes that need them, informing all
  // sharing processes of their destinations
  std::map<std::int32_t, std::set<std::uint32_t>> shared_vertices;
  EigenRowArrayXXd vertex_coordinates(vertex_indices.size(),
                                      mesh_data.geometry.dim);

  distribute_vertices(comm, mesh_data, vertex_indices, vertex_coordinates,
                      shared_vertices);

  timer.stop();

  // Build local mesh from new_mesh_data
  mesh::Mesh mesh = build_local_mesh(
      comm, new_global_cell_indices, local_cell_vertices,
      mesh_data.topology.cell_type, mesh_data.topology.dim,
      mesh_data.topology.num_global_cells, vertex_indices, vertex_coordinates,
      mesh_data.geometry.dim, mesh_data.geometry.num_global_vertices);

  // Fix up some of the ancilliary data about sharing and ownership
  // now that the mesh has been initialised

  // Copy cell ownership
  std::vector<std::uint32_t>& cell_owner = mesh.topology().cell_owner();
  cell_owner.clear();
  cell_owner.insert(cell_owner.begin(),
                    new_cell_partition.begin() + num_regular_cells,
                    new_cell_partition.end());

  // Set the ghost cell offset
  mesh.topology().init_ghost(tdim, num_regular_cells);

  // Set the ghost vertex offset
  mesh.topology().init_ghost(0, num_regular_vertices);

  // Assign map of shared cells and vertices
  mesh.topology().shared_entities(mesh_data.topology.dim) = shared_cells;
  mesh.topology().shared_entities(0) = shared_vertices;

  return mesh;
}
//-----------------------------------------------------------------------------
void MeshPartitioning::reorder_cells_gps(
    MPI_Comm mpi_comm, const std::uint32_t num_regular_cells,
    const CellType& cell_type,
    const std::map<std::int32_t, std::set<std::uint32_t>>& shared_cells,
    const boost::multi_array<std::int64_t, 2>& cell_vertices,
    const std::vector<std::int64_t>& global_cell_indices,
    std::map<std::int32_t, std::set<std::uint32_t>>& reordered_shared_cells,
    boost::multi_array<std::int64_t, 2>& reordered_cell_vertices,
    std::vector<std::int64_t>& reordered_global_cell_indices)
{
  log::log(PROGRESS, "Re-order cells during distributed mesh construction");

  common::Timer timer("Reorder cells using GPS ordering");

  // Make dual graph from vertex indices, using GraphBuilder
  // FIXME: this should be reused later to add the facet-cell topology
  std::vector<std::vector<std::size_t>> local_graph;
  dolfin::graph::GraphBuilder::FacetCellMap facet_cell_map;
  dolfin::graph::GraphBuilder::compute_local_dual_graph(
      mpi_comm, cell_vertices, cell_type, local_graph, facet_cell_map);

  const std::size_t num_all_cells = cell_vertices.shape()[0];
  const std::size_t local_cell_offset
      = MPI::global_offset(mpi_comm, num_all_cells, true);

  // Convert between graph types, removing offset
  // FIXME: make all graphs the same type

  dolfin::graph::Graph g_dual;
  // Ignore the ghost cells - they will not be reordered
  // FIXME: reorder ghost cells too
  for (std::uint32_t i = 0; i != num_regular_cells; ++i)
  {
    dolfin::common::Set<int> conn_set;
    for (auto q = local_graph[i].begin(); q != local_graph[i].end(); ++q)
    {
      dolfin_assert(*q >= local_cell_offset);
      const int local_index = *q - local_cell_offset;

      // Ignore ghost cells in connectivity
      if (local_index < (int)num_regular_cells)
        conn_set.insert(local_index);
    }
    g_dual.push_back(conn_set);
  }
  std::vector<int> remap = dolfin::graph::SCOTCH::compute_gps(g_dual);

  // Resize re-ordered cell topology arrray, and copy (copy iss
  // required because ghosts are not being re-ordered).
  reordered_cell_vertices.resize(
      boost::extents[cell_vertices.shape()[0]][cell_vertices.shape()[1]]);
  reordered_cell_vertices = cell_vertices;

  // Assign old gloabl indeices to re-ordered indices (since ghost
  // will be be re-ordered, we need to copy them over)
  reordered_global_cell_indices = global_cell_indices;

  for (std::uint32_t i = 0; i != g_dual.size(); ++i)
  {
    // Remap data
    const std::uint32_t j = remap[i];
    reordered_cell_vertices[j] = cell_vertices[i];
    reordered_global_cell_indices[j] = global_cell_indices[i];
  }

  // Clear data
  reordered_shared_cells.clear();
  for (auto p = shared_cells.begin(); p != shared_cells.end(); ++p)
  {
    const std::uint32_t cell_index = p->first;
    if (cell_index < num_regular_cells)
      reordered_shared_cells.insert({remap[cell_index], p->second});
    else
      reordered_shared_cells.insert(*p);
  }
}
//-----------------------------------------------------------------------------
void MeshPartitioning::distribute_cell_layer(
    MPI_Comm mpi_comm, const int num_regular_cells,
    const std::int64_t num_global_vertices,
    std::map<std::int32_t, std::set<std::uint32_t>>& shared_cells,
    boost::multi_array<std::int64_t, 2>& cell_vertices,
    std::vector<std::int64_t>& global_cell_indices,
    std::vector<int>& cell_partition)
{
  common::Timer timer("Distribute cell layer");

  const int mpi_size = MPI::size(mpi_comm);
  const int mpi_rank = MPI::rank(mpi_comm);

  // Get set of vertices in ghost cells
  std::map<std::int64_t, std::vector<std::int64_t>> sh_vert_to_cell;

  // Make global-to-local map of shared cells
  std::map<std::int64_t, int> cell_global_to_local;
  for (int i = num_regular_cells; i < (int)cell_vertices.size(); ++i)
  {
    // Add map entry for each vertex
    for (auto p = cell_vertices[i].begin(); p != cell_vertices[i].end(); ++p)
      sh_vert_to_cell.insert({*p, std::vector<std::int64_t>()});

    cell_global_to_local.insert({global_cell_indices[i], i});
  }

  // Reduce vertex set to those which also appear in local cells
  // giving the effective boundary vertices.  Make a map from these
  // vertices to the set of connected cells (but only adding locally
  // owned cells)

  // Go through all regular cells to add any previously unshared
  // cells.
  for (int i = 0; i < num_regular_cells; ++i)
  {
    for (auto v = cell_vertices[i].begin(); v != cell_vertices[i].end(); ++v)
    {
      auto vc_it = sh_vert_to_cell.find(*v);
      if (vc_it != sh_vert_to_cell.end())
      {
        cell_global_to_local.insert({global_cell_indices[i], i});
        vc_it->second.push_back(i);
      }
    }
  }

  // Send lists of cells/owners to MPI::index_owner of vertex,
  // collating and sending back out...
  std::vector<std::vector<std::int64_t>> send_vertcells(mpi_size);
  std::vector<std::vector<std::int64_t>> recv_vertcells(mpi_size);
  for (auto vc_it = sh_vert_to_cell.begin(); vc_it != sh_vert_to_cell.end();
       ++vc_it)
  {
    const int dest
        = MPI::index_owner(mpi_comm, vc_it->first, num_global_vertices);

    std::vector<std::int64_t>& sendv = send_vertcells[dest];

    // Pack as [cell_global_index, this_vertex, [other_vertices]]
    for (auto q = vc_it->second.begin(); q != vc_it->second.end(); ++q)
    {
      sendv.push_back(global_cell_indices[*q]);
      sendv.push_back(vc_it->first);
      for (auto v = cell_vertices[*q].begin(); v != cell_vertices[*q].end();
           ++v)
      {
        if (*v != vc_it->first)
          sendv.push_back(*v);
      }
    }
  }

  MPI::all_to_all(mpi_comm, send_vertcells, recv_vertcells);

  const std::uint32_t num_cell_vertices = cell_vertices.shape()[1];

  // Collect up cells on common vertices

  // Reset map
  sh_vert_to_cell.clear();
  for (int i = 0; i < mpi_size; ++i)
  {
    const std::vector<std::int64_t>& recv_i = recv_vertcells[i];
    for (auto q = recv_i.begin(); q != recv_i.end(); q += num_cell_vertices + 1)
    {
      const std::size_t vertex_index = *(q + 1);
      std::vector<std::int64_t> cell_set = {i};
      cell_set.insert(cell_set.end(), q, q + num_cell_vertices + 1);

      // Packing: [owner, cell_index, this_vertex, [other_vertices]]
      // Look for vertex in map, and add the attached cell
      auto it = sh_vert_to_cell.find(vertex_index);
      if (it == sh_vert_to_cell.end())
        sh_vert_to_cell.insert({vertex_index, cell_set});
      else
        it->second.insert(it->second.end(), cell_set.begin(), cell_set.end());
    }
  }

  // Clear sending arrays
  send_vertcells = std::vector<std::vector<std::int64_t>>(mpi_size);

  // Send back out to all processes which share the same vertex
  // FIXME: avoid sending back own cells to owner?
  for (auto p = sh_vert_to_cell.begin(); p != sh_vert_to_cell.end(); ++p)
  {
    for (auto q = p->second.begin(); q != p->second.end();
         q += (num_cell_vertices + 2))
    {
      send_vertcells[*q].insert(send_vertcells[*q].end(), p->second.begin(),
                                p->second.end());
    }
  }

  MPI::all_to_all(mpi_comm, send_vertcells, recv_vertcells);

  // Count up new cells, assign local index, set owner
  // and initialise shared_cells

  const std::uint32_t num_cells = cell_vertices.shape()[0];
  std::uint32_t count = num_cells;

  for (auto p = recv_vertcells.begin(); p != recv_vertcells.end(); ++p)
  {
    for (auto q = p->begin(); q != p->end(); q += num_cell_vertices + 2)
    {
      const std::int64_t owner = *q;
      const std::int64_t cell_index = *(q + 1);
      auto cell_it = cell_global_to_local.find(cell_index);
      if (cell_it == cell_global_to_local.end())
      {
        cell_global_to_local.insert({cell_index, count});
        shared_cells.insert({count, std::set<std::uint32_t>()});
        global_cell_indices.push_back(cell_index);
        cell_partition.push_back(owner);
        ++count;
      }
    }
  }

  cell_vertices.resize(boost::extents[count][num_cell_vertices]);
  std::set<std::uint32_t> sharing_procs;
  std::vector<std::size_t> sharing_cells;
  std::size_t last_vertex = std::numeric_limits<std::size_t>::max();
  for (auto p = recv_vertcells.begin(); p != recv_vertcells.end(); ++p)
  {
    for (auto q = p->begin(); q != p->end(); q += num_cell_vertices + 2)
    {
      const std::size_t shared_vertex = *(q + 2);
      const int owner = *q;
      const std::size_t cell_index = *(q + 1);
      const std::size_t local_index
          = cell_global_to_local.find(cell_index)->second;

      // Add vertices to new cells
      if (local_index >= num_cells)
      {
        for (std::uint32_t j = 0; j != num_cell_vertices; ++j)
          cell_vertices[local_index][j] = *(q + j + 2);
      }

      // If starting on a new shared vertex, dump old data into
      // shared_cells
      if (shared_vertex != last_vertex)
      {
        last_vertex = shared_vertex;
        for (auto c = sharing_cells.begin(); c != sharing_cells.end(); ++c)
        {
          auto it = shared_cells.find(*c);
          if (it == shared_cells.end())
            shared_cells.insert({*c, sharing_procs});
          else
            it->second.insert(sharing_procs.begin(), sharing_procs.end());
        }
        sharing_procs.clear();
        sharing_cells.clear();
      }

      // Don't include self in sharing processes
      if (owner != mpi_rank)
        sharing_procs.insert(owner);
      sharing_cells.push_back(local_index);
    }
  }

  for (auto c = sharing_cells.begin(); c != sharing_cells.end(); ++c)
  {
    auto it = shared_cells.find(*c);
    if (it == shared_cells.end())
      shared_cells.insert({*c, sharing_procs});
    else
      it->second.insert(sharing_procs.begin(), sharing_procs.end());
  }

  // Shrink
  global_cell_indices.shrink_to_fit();
  cell_partition.shrink_to_fit();
}
//-----------------------------------------------------------------------------
std::int32_t MeshPartitioning::distribute_cells(
    const MPI_Comm mpi_comm, const LocalMeshData& mesh_data,
    const std::vector<int>& cell_partition,
    const std::map<std::int64_t, std::vector<int>>& ghost_procs,
    boost::multi_array<std::int64_t, 2>& new_cell_vertices,
    std::vector<std::int64_t>& new_global_cell_indices,
    std::vector<int>& new_cell_partition,
    std::map<std::int32_t, std::set<std::uint32_t>>& shared_cells)
{
  // This function takes the partition computed by the partitioner
  // stored in cell_partition/ghost_procs Some cells go to multiple
  // destinations. Each cell is transmitted to its final
  // destination(s) including its global index, and the cell owner
  // (for ghost cells this will be different from the destination)

  log::log(PROGRESS, "Distribute cells during distributed mesh construction");

  common::Timer timer("Distribute cells");

  const std::size_t mpi_size = MPI::size(mpi_comm);
  const std::size_t mpi_rank = MPI::rank(mpi_comm);

  new_cell_partition.clear();
  shared_cells.clear();

  // Get dimensions of local mesh_data
  const std::size_t num_local_cells = mesh_data.topology.cell_vertices.size();
  dolfin_assert(mesh_data.topology.global_cell_indices.size()
                == num_local_cells);
  const std::size_t num_cell_vertices
      = mesh_data.topology.num_vertices_per_cell;
  if (!mesh_data.topology.cell_vertices.empty())
  {
    if (mesh_data.topology.cell_vertices[0].size() != num_cell_vertices)
    {
      log::dolfin_error(
          "MeshPartitioning.cpp", "distribute cells",
          "Mismatch in number of cell vertices (%d != %d) on process %d",
          mesh_data.topology.cell_vertices[0].size(), num_cell_vertices,
          mpi_rank);
    }
  }

  // Send all cells to their destinations including their global
  // indices.  First element of vector is cell count of unghosted
  // cells, second element is count of ghost cells.
  std::vector<std::vector<std::size_t>> send_cell_vertices(
      mpi_size, std::vector<std::size_t>(2, 0));

  for (std::uint32_t i = 0; i != cell_partition.size(); ++i)
  {
    // If cell is in ghost_procs map, use that to determine
    // destinations, otherwise just use the cell_partition vector
    auto map_it = ghost_procs.find(i);
    if (map_it != ghost_procs.end())
    {
      const std::vector<int>& destinations = map_it->second;
      for (auto dest = destinations.begin(); dest != destinations.end(); ++dest)
      {
        // Create reference to destination vector
        std::vector<std::size_t>& send_cell_dest = send_cell_vertices[*dest];

        // Count of ghost cells, followed by ghost processes
        send_cell_dest.push_back(destinations.size());
        send_cell_dest.insert(send_cell_dest.end(), destinations.begin(),
                              destinations.end());

        // Global cell index
        send_cell_dest.push_back(mesh_data.topology.global_cell_indices[i]);

        // Global vertex indices
        send_cell_dest.insert(send_cell_dest.end(),
                              mesh_data.topology.cell_vertices[i].begin(),
                              mesh_data.topology.cell_vertices[i].end());

        // First entry is the owner, so this counts as a 'local' cell
        // subsequent entries are 'remote ghosts'
        if (dest == destinations.begin())
          send_cell_dest[0]++;
        else
          send_cell_dest[1]++;
      }
    }
    else
    {
      // Single destination (unghosted cell)
      std::vector<std::size_t>& send_cell_dest
          = send_cell_vertices[cell_partition[i]];
      send_cell_dest.push_back(0);

      // Global cell index
      send_cell_dest.push_back(mesh_data.topology.global_cell_indices[i]);

      // Global vertex indices
      send_cell_dest.insert(send_cell_dest.end(),
                            mesh_data.topology.cell_vertices[i].begin(),
                            mesh_data.topology.cell_vertices[i].end());
      send_cell_dest[0]++;
    }
  }

  // Distribute cell-vertex connectivity and ownership information
  std::vector<std::vector<std::size_t>> received_cell_vertices(mpi_size);
  MPI::all_to_all(mpi_comm, send_cell_vertices, received_cell_vertices);

  // Count number of received cells (first entry in vector) and find
  // out how many ghost cells there are...
  std::size_t local_count = 0;
  std::size_t ghost_count = 0;
  for (std::size_t p = 0; p < mpi_size; ++p)
  {
    std::vector<std::size_t>& received_data = received_cell_vertices[p];
    local_count += received_data[0];
    ghost_count += received_data[1];
  }

  const std::size_t all_count = ghost_count + local_count;

  // Put received mesh data into new_mesh_data structure
  new_cell_vertices.resize(boost::extents[all_count][num_cell_vertices]);
  new_global_cell_indices.resize(all_count);
  new_cell_partition.resize(all_count);

  // Unpack received data
  // Create a map from cells which are shared, to the remote processes
  // which share them - corral ghost cells to end of range
  std::size_t c = 0;
  std::size_t gc = local_count;
  for (std::size_t p = 0; p < mpi_size; ++p)
  {
    std::vector<std::size_t>& received_data = received_cell_vertices[p];
    for (auto it = received_data.begin() + 2; it != received_data.end();
         it += (*it + num_cell_vertices + 2))
    {
      auto tmp_it = it;
      const std::uint32_t num_ghosts = *tmp_it++;

      // Determine owner, and indexing.
      // Note that *tmp_it may be equal to mpi_rank
      const std::size_t owner = (num_ghosts == 0) ? mpi_rank : *tmp_it;
      const std::size_t idx = (owner == mpi_rank) ? c : gc;

      dolfin_assert(idx < new_cell_partition.size());
      new_cell_partition[idx] = owner;
      if (num_ghosts != 0)
      {
        std::set<std::uint32_t> proc_set(tmp_it, tmp_it + num_ghosts);

        // Remove self from set of sharing processes
        proc_set.erase(mpi_rank);
        shared_cells.insert({idx, proc_set});
        tmp_it += num_ghosts;
      }

      new_global_cell_indices[idx] = *tmp_it++;
      for (std::size_t j = 0; j < num_cell_vertices; ++j)
        new_cell_vertices[idx][j] = *tmp_it++;

      if (owner == mpi_rank)
        ++c;
      else
        ++gc;
    }
  }

  dolfin_assert(c == local_count);
  dolfin_assert(gc == all_count);
  return local_count;
}
//-----------------------------------------------------------------------------
std::int32_t MeshPartitioning::compute_vertex_mapping(
    MPI_Comm mpi_comm, const std::int32_t num_regular_cells,
    const boost::multi_array<std::int64_t, 2>& cell_vertices,
    std::vector<std::int64_t>& vertex_indices,
    Eigen::Ref<EigenRowArrayXXi32> local_cell_vertices)
{
  std::map<std::int64_t, std::int32_t> vertex_global_to_local;
  vertex_indices.clear();
  vertex_global_to_local.clear();

  const std::int32_t num_cells = cell_vertices.size();
  const std::int32_t num_cell_vertices = cell_vertices.shape()[1];

  local_cell_vertices.resize(num_cells, num_cell_vertices);

  // Get set of unique vertices from cells and start constructing a
  // global_to_local map.  Ghost vertices will be at the end of the
  // range (v >= num_regular_vertices).
  std::int32_t v = 0;
  std::int32_t num_regular_vertices = 0;
  for (std::int32_t i = 0; i < num_cells; ++i)
  {
    for (std::int32_t j = 0; j < num_cell_vertices; ++j)
    {
      std::int64_t q = cell_vertices[i][j];
      auto map_it = vertex_global_to_local.insert({q, v});
      local_cell_vertices(i, j) = map_it.first->second;
      if (map_it.second)
      {
        vertex_indices.push_back(q);
        ++v;
        if (i < num_regular_cells)
          num_regular_vertices = v;
      }
    }
  }

  return num_regular_vertices;
}
//-----------------------------------------------------------------------------
void MeshPartitioning::distribute_vertices(
    const MPI_Comm mpi_comm, const LocalMeshData& mesh_data,
    const std::vector<std::int64_t>& vertex_indices,
    Eigen::Ref<EigenRowArrayXXd> vertex_coordinates,
    std::map<std::int32_t, std::set<std::uint32_t>>& shared_vertices_local)
{
  // This function distributes all vertices (coordinates and
  // local-to-global mapping) according to the cells that are stored
  // on each process. This happens in several stages: First each
  // process figures out which vertices it needs (by looking at its
  // cells) and where those vertices are located. That information is
  // then distributed so that each process learns where it needs to
  // send its vertices.

  log::log(PROGRESS,
           "Distribute vertices during distributed mesh construction");
  common::Timer timer("Distribute vertices");

  // Get number of processes
  const int mpi_size = MPI::size(mpi_comm);
  const int mpi_rank = MPI::rank(mpi_comm);

  // Get geometric dimension
  const int gdim = mesh_data.geometry.dim;

  // Compute where (process number) the vertices we need are located
  std::vector<std::size_t> ranges(mpi_size);
  MPI::all_gather(mpi_comm, mesh_data.geometry.vertex_indices.size(), ranges);
  for (std::uint32_t i = 1; i != ranges.size(); ++i)
    ranges[i] += ranges[i - 1];
  ranges.insert(ranges.begin(), 0);

  std::vector<std::vector<std::size_t>> send_vertex_indices(mpi_size);
  std::vector<std::vector<std::uint32_t>> local_indexing(mpi_size);
  for (unsigned int i = 0; i != vertex_indices.size(); ++i)
  {
    const std::size_t required_vertex = vertex_indices[i];
    const int location
        = std::upper_bound(ranges.begin(), ranges.end(), required_vertex)
          - ranges.begin() - 1;
    send_vertex_indices[location].push_back(required_vertex);
    local_indexing[location].push_back(i);
  }

  // Piggy-back local offset onto end of sending arrays
  std::size_t offset = 0;
  for (int i = 0; i != mpi_size; ++i)
  {
    send_vertex_indices[i].push_back(offset);
    offset += (send_vertex_indices[i].size() - 1);
  }

  // Send required vertices to other processes, and receive
  // vertices required by other processes.
  std::vector<std::vector<std::size_t>> received_vertex_indices;
  MPI::all_to_all(mpi_comm, send_vertex_indices, received_vertex_indices);

  // Extract remote offsets for sending data with MPI_Put
  std::vector<std::size_t> remote_offsets;
  std::size_t num_received_indices = 0;
  for (auto& p : received_vertex_indices)
  {
    remote_offsets.push_back(p.back());
    p.pop_back();
    num_received_indices += p.size();
  }

  // Pop offset off back of sending arrays too
  for (auto& p : send_vertex_indices)
    p.pop_back();

  // Array to receive data into with RMA
  EigenRowArrayXXd receive_coord_data(vertex_indices.size(), gdim);

  // Create local RMA window
  MPI_Win win;
  MPI_Win_create(receive_coord_data.data(),
                 sizeof(double) * vertex_indices.size() * gdim, sizeof(double),
                 MPI_INFO_NULL, mpi_comm, &win);
  MPI_Win_fence(0, win);

  // Put data to remote with RMA
  EigenRowArrayXXd send_coord_data(num_received_indices, gdim);
  Eigen::Map<const EigenRowArrayXXd> mesh_data_vertices(
      mesh_data.geometry.vertex_coordinates.data(),
      mesh_data.geometry.vertex_coordinates.shape()[0],
      mesh_data.geometry.vertex_coordinates.shape()[1]);

  const std::pair<std::size_t, std::size_t> local_vertex_range
      = {ranges[mpi_rank], ranges[mpi_rank + 1]};

  std::size_t local_index = 0;
  for (int p = 0; p < mpi_size; ++p)
  {
    const std::size_t local_index_0 = local_index;
    for (const auto& q : received_vertex_indices[p])
    {
      dolfin_assert(q >= local_vertex_range.first
                    && q < local_vertex_range.second);

      const std::size_t location = q - local_vertex_range.first;
      send_coord_data.row(local_index) = mesh_data_vertices.row(location);
      ++local_index;
    }

    const std::size_t local_size = (local_index - local_index_0) * gdim;
    MPI_Put(send_coord_data.data() + local_index_0 * gdim, local_size,
            MPI_DOUBLE, p, remote_offsets[p] * gdim, local_size, MPI_DOUBLE,
            win);
  }

  // Meanwhile, redistribute received_vertex_indices as vertex sharing
  // information
  build_shared_vertices(mpi_comm, shared_vertices_local,
                        received_vertex_indices, local_vertex_range,
                        local_indexing);

  // Synchronise and free RMA window
  MPI_Win_fence(0, win);
  MPI_Win_free(&win);

  // Reorder coordinates according to global_to_local mapping
  local_index = 0;
  for (int p = 0; p < mpi_size; ++p)
  {
    for (const auto& v : local_indexing[p])
    {
      vertex_coordinates.row(v) = receive_coord_data.row(local_index);
      ++local_index;
    }
  }
}
//-----------------------------------------------------------------------------
void MeshPartitioning::build_shared_vertices(
    MPI_Comm mpi_comm,
    std::map<std::int32_t, std::set<std::uint32_t>>& shared_vertices_local,
    const std::vector<std::vector<std::size_t>>& received_vertex_indices,
    const std::pair<std::size_t, std::size_t> local_vertex_range,
    const std::vector<std::vector<std::uint32_t>>& local_indexing)
{
  log::log(PROGRESS,
           "Build shared vertices during distributed mesh construction");

  const std::uint32_t mpi_size = MPI::size(mpi_comm);

  // Count number sharing each local vertex
  std::vector<std::int32_t> n_sharing(
      local_vertex_range.second - local_vertex_range.first, 0);
  for (std::uint32_t p = 0; p < mpi_size; ++p)
    for (const auto& q : received_vertex_indices[p])
    {
      dolfin_assert(q >= local_vertex_range.first
                    and q < local_vertex_range.second);
      const std::size_t local_index = q - local_vertex_range.first;
      ++n_sharing[local_index];
    }

  // Create an array of 'pointers' to shared entries (where p > 1)
  // Set to 0 for unshared entries
  std::vector<std::int32_t> offset;
  offset.reserve(n_sharing.size());
  std::int32_t index = 0;
  for (auto& p : n_sharing)
  {
    if (p == 1)
      p = 0;

    offset.push_back(index);
    index += (p * 2);
  }

  // Fill with list of sharing processes and position in
  // received_vertex_indices to send back to originating process
  std::vector<std::int32_t> process_list(index);
  for (std::uint32_t p = 0; p < mpi_size; ++p)
    for (unsigned int i = 0; i < received_vertex_indices[p].size(); ++i)
    {
      const std::size_t q = received_vertex_indices[p][i];
      const std::size_t local_index = q - local_vertex_range.first;
      std::int32_t& location = offset[local_index];
      if (n_sharing[local_index] > 0)
      {
        process_list[location] = p;
        ++location;
        process_list[location] = i;
        ++location;
      }
    }

  // Reset offsets to original positions
  for (unsigned int i = 0; i != offset.size(); ++i)
    offset[i] -= 2 * n_sharing[i];

  std::vector<std::vector<std::size_t>> send_sharing(mpi_size);
  for (unsigned int i = 0; i != n_sharing.size(); ++i)
  {
    if (n_sharing[i] > 0)
    {
      for (int j = 0; j < n_sharing[i]; ++j)
      {
        auto& ss = send_sharing[process_list[offset[i] + j * 2]];
        ss.push_back(n_sharing[i] - 1);
        ss.push_back(process_list[offset[i] + j * 2 + 1]);
        for (int k = 0; k < n_sharing[i]; ++k)
          if (j != k)
            ss.push_back(process_list[offset[i] + k * 2]);
      }
    }
  }

  // Receive sharing information back to original processes
  std::vector<std::vector<std::size_t>> recv_sharing(mpi_size);
  MPI::all_to_all(mpi_comm, send_sharing, recv_sharing);

  for (unsigned int p = 0; p < mpi_size; ++p)
  {
    const std::vector<std::uint32_t>& local_index_p = local_indexing[p];
    for (auto q = recv_sharing[p].begin(); q != recv_sharing[p].end();
         q += (*q + 2))
    {
      const std::size_t num_sharing = *q;
      const std::uint32_t local_index = local_index_p[*(q + 1)];
      std::set<std::uint32_t> sharing_processes(q + 2, q + 2 + num_sharing);

      dolfin_assert(shared_vertices_local.find(local_index)
                    == shared_vertices_local.end());
      shared_vertices_local.insert({local_index, sharing_processes});
    }
  }
}
//-----------------------------------------------------------------------------
mesh::Mesh MeshPartitioning::build_local_mesh(
    const MPI_Comm& comm, const std::vector<std::int64_t>& global_cell_indices,
    Eigen::Ref<const EigenRowArrayXXi32> cells,
    const mesh::CellType::Type cell_type, const int tdim,
    const std::int64_t num_global_cells,
    const std::vector<std::int64_t>& vertex_indices,
    Eigen::Ref<const EigenRowArrayXXd> vertex_coordinates, const int gdim,
    const std::int64_t num_global_vertices)
{
  log::log(PROGRESS, "Build local mesh during distributed mesh construction");
  common::Timer timer(
      "Build local part of distributed mesh (from local mesh data)");

  mesh::Mesh mesh(comm, cell_type, vertex_coordinates, cells);

  // Reset global indices
  const std::size_t num_vertices = vertex_coordinates.rows();
  const std::size_t num_cells = cells.rows();
  mesh.topology().init(0, num_vertices, num_global_vertices);
  mesh.topology().init(tdim, num_cells, num_global_cells);

  // Set global indices for vertices
  for (std::size_t i = 0; i < vertex_indices.size(); ++i)
    mesh.topology().set_global_index(0, i, vertex_indices[i]);

  // Set global indices for cells
  for (std::size_t i = 0; i < global_cell_indices.size(); ++i)
    mesh.topology().set_global_index(tdim, i, global_cell_indices[i]);

  return mesh;
}
//-----------------------------------------------------------------------------
