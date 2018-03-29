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
mesh::Mesh
MeshPartitioning::build_distributed_mesh(const LocalMeshData& local_data,
                                         const std::string ghost_mode)
{
  log::log(PROGRESS, "Building distributed mesh");

  common::Timer timer("Build distributed mesh from local mesh data");

  // MPI communicator
  MPI_Comm comm = local_data.mpi_comm();

  Eigen::Map<const EigenRowArrayXXi64> cells(
      local_data.topology.cell_vertices.data(),
      local_data.topology.cell_vertices.shape()[0],
      local_data.topology.cell_vertices.shape()[1]);

  Eigen::Map<const EigenRowArrayXXd> points(
      local_data.geometry.vertex_coordinates.data(),
      local_data.geometry.vertex_coordinates.shape()[0],
      local_data.geometry.vertex_coordinates.shape()[1]);

  mesh::CellType::Type type = local_data.topology.cell_type;

  return build_distributed_mesh(comm, type, points, cells,
                                local_data.topology.global_cell_indices,
                                ghost_mode);
}
//-----------------------------------------------------------------------------
mesh::Mesh MeshPartitioning::build_distributed_mesh(
    const MPI_Comm& comm, mesh::CellType::Type type,
    Eigen::Ref<const EigenRowArrayXXd> points,
    Eigen::Ref<const EigenRowArrayXXi64> cells,
    const std::vector<std::int64_t>& global_cell_indices,
    const std::string ghost_mode)
{

  // Get mesh partitioner
  const std::string partitioner = parameter::parameters["mesh_partitioner"];
  // Compute the cell partition
  MeshPartition mp = partition_cells(comm, type, cells, partitioner);

  // Check that we have some ghost information.
  int all_ghosts = MPI::sum(comm, mp.num_ghosts());
  if (all_ghosts == 0 && ghost_mode != "none")
  {
    log::dolfin_error("MeshPartitioning.cpp", "build ghost mesh",
                      "Ghost cell information not available");
  }

  // Build mesh from local mesh data and provided cell partition
  mesh::Mesh mesh
      = build(comm, type, cells, points, global_cell_indices, ghost_mode, mp);

  // Store used ghost mode
  // NOTE: This is the only place in DOLFIN which eventually sets
  //       mesh._ghost_mode != "none"
  mesh._ghost_mode = ghost_mode;
  mesh.order();

  // Initialise number of globally connected cells to each facet. This
  // is necessary to distinguish between facets on an exterior
  // boundary and facets on a partition boundary (see
  // https://bugs.launchpad.net/dolfin/+bug/733834).

  DistributedMeshTools::init_facet_cell_connections(mesh);

  return mesh;
}
//-----------------------------------------------------------------------------
MeshPartition MeshPartitioning::partition_cells(
    const MPI_Comm& mpi_comm, mesh::CellType::Type type,
    Eigen::Ref<const EigenRowArrayXXi64> cell_vertices,
    const std::string partitioner)
{
  log::log(PROGRESS, "Compute partition of cells across processes");

  std::unique_ptr<mesh::CellType> cell_type(mesh::CellType::create(type));
  dolfin_assert(cell_type);

  // Compute cell partition using partitioner from parameter system
  if (partitioner == "SCOTCH")
  {
    return dolfin::graph::SCOTCH::compute_partition(mpi_comm, cell_vertices,
                                                    *cell_type);
  }
  else if (partitioner == "ParMETIS")
  {
    return dolfin::graph::ParMETIS::compute_partition(mpi_comm, cell_vertices,
                                                      *cell_type);
  }
  else
  {
    log::dolfin_error("MeshPartitioning.cpp", "compute cell partition",
                      "Mesh partitioner '%s' is unknown.", partitioner.c_str());
  }
  return MeshPartition({}, {});
}
//-----------------------------------------------------------------------------
mesh::Mesh
MeshPartitioning::build(const MPI_Comm& comm, mesh::CellType::Type type,
                        Eigen::Ref<const EigenRowArrayXXi64> cell_vertices,
                        Eigen::Ref<const EigenRowArrayXXd> points,
                        const std::vector<std::int64_t>& global_cell_indices,
                        const std::string ghost_mode, const MeshPartition& mp)
{
  // Distribute cells
  log::log(PROGRESS, "Distribute mesh cells");

  common::Timer timer("Distribute mesh cells");

  // Create CellType objects based on current cell type
  std::unique_ptr<mesh::CellType> cell_type(mesh::CellType::create(type));
  dolfin_assert(cell_type);

  // Topological dimension
  const int tdim = cell_type->dim();

  const std::int64_t num_global_vertices = MPI::sum(comm, points.rows());
  const std::int64_t num_global_cells = MPI::sum(comm, cell_vertices.rows());

  EigenRowArrayXXi64 new_cell_vertices;
  std::vector<std::int64_t> new_global_cell_indices;
  std::vector<int> new_cell_partition;
  std::map<std::int32_t, std::set<std::uint32_t>> shared_cells;

  // Send cells to owning process according to mp cell partition, and
  // receive cells that belong to this process. Also compute auxiliary
  // data related to sharing.

  const std::int32_t num_regular_cells = distribute_cells(
      comm, cell_vertices, global_cell_indices, mp, new_cell_vertices,
      new_global_cell_indices, new_cell_partition, shared_cells);

  if (ghost_mode == "shared_vertex")
  {
    log::dolfin_error("MeshPartitioning.cpp", "use shared_vertex mode",
                      "Needs fixing");

    // Send/receive additional cells defined by connectivity to the shared
    // vertices.
    //    distribute_cell_layer(comm, num_regular_cells, num_global_vertices,
    //                          shared_cells, new_cell_vertices,
    //                          new_global_cell_indices, new_cell_partition);
  }
  else if (ghost_mode == "none")
  {
    // Resize to remove all ghost cells
    new_cell_partition.resize(num_regular_cells);
    new_global_cell_indices.resize(num_regular_cells);
    new_cell_vertices.conservativeResize(num_regular_cells, Eigen::NoChange);
    shared_cells.clear();
  }

#ifdef HAS_SCOTCH
  if (parameter::parameters["reorder_cells_gps"])
  {
    // Allocate objects to hold re-ordering
    std::map<std::int32_t, std::set<std::uint32_t>> reordered_shared_cells;
    EigenRowArrayXXi64 reordered_cell_vertices(new_cell_vertices.rows(),
                                               new_cell_vertices.cols());
    std::vector<std::int64_t> reordered_global_cell_indices;

    // Re-order cells
    reorder_cells_gps(comm, num_regular_cells, *cell_type, shared_cells,
                      new_cell_vertices, new_global_cell_indices,
                      reordered_shared_cells, reordered_cell_vertices,
                      reordered_global_cell_indices);

    // Update to re-ordered indices
    std::swap(shared_cells, reordered_shared_cells);
    new_cell_vertices = reordered_cell_vertices;
    std::swap(new_global_cell_indices, reordered_global_cell_indices);
  }
#endif

  timer.stop();

  // Build mesh from points and distributed cells
  mesh::Mesh mesh(comm, type, points, new_cell_vertices);

  // Reset global indices
  const std::size_t num_vertices = mesh.num_entities(0);
  const std::size_t num_cells = mesh.num_cells();
  mesh.topology().init(0, num_vertices, num_global_vertices);
  mesh.topology().init(tdim, num_cells, num_global_cells);

  // Set global indices for cells
  for (std::size_t i = 0; i < new_global_cell_indices.size(); ++i)
    mesh.topology().set_global_index(tdim, i, new_global_cell_indices[i]);

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

  // Find highest index + 1 in local_cell_vertices of regular cells
  std::int32_t num_regular_vertices
      = *std::max_element(mesh.topology()(tdim, 0)().data(),
                          mesh.topology()(tdim, 0)().data()
                              + new_cell_vertices.cols() * num_regular_cells)
        + 1;

  // Set the ghost vertex offset
  mesh.topology().init_ghost(0, num_regular_vertices);

  // Assign map of shared cells and vertices
  mesh.topology().shared_entities(tdim) = shared_cells;

  return mesh;
}
//-----------------------------------------------------------------------------
void MeshPartitioning::reorder_cells_gps(
    MPI_Comm mpi_comm, const std::uint32_t num_regular_cells,
    const CellType& cell_type,
    const std::map<std::int32_t, std::set<std::uint32_t>>& shared_cells,
    Eigen::Ref<EigenRowArrayXXi64> global_cell_vertices,
    const std::vector<std::int64_t>& global_cell_indices,
    std::map<std::int32_t, std::set<std::uint32_t>>& reordered_shared_cells,
    Eigen::Ref<EigenRowArrayXXi64> reordered_cell_vertices,
    std::vector<std::int64_t>& reordered_global_cell_indices)
{
  log::log(PROGRESS, "Re-order cells during distributed mesh construction");

  common::Timer timer("Reorder cells using GPS ordering");

  // Make dual graph from vertex indices, using GraphBuilder
  // FIXME: this should be reused later to add the facet-cell topology
  std::vector<std::vector<std::size_t>> local_graph;
  dolfin::graph::GraphBuilder::FacetCellMap facet_cell_map;

  dolfin::graph::GraphBuilder::compute_local_dual_graph(
      mpi_comm, global_cell_vertices, cell_type, local_graph, facet_cell_map);

  const std::size_t num_all_cells = global_cell_vertices.rows();
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

  // Add direct mapping for any ghost cells (not reordered)
  for (unsigned int j = remap.size(); j < global_cell_indices.size(); ++j)
    remap.push_back(j);

  for (std::uint32_t i = 0; i != g_dual.size(); ++i)
  {
    // Remap data
    const std::uint32_t j = remap[i];
    reordered_cell_vertices.row(j) = global_cell_vertices.row(i);
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
    const MPI_Comm mpi_comm, Eigen::Ref<const EigenRowArrayXXi64> cell_vertices,
    const std::vector<std::int64_t>& global_cell_indices,
    const MeshPartition& mp, EigenRowArrayXXi64& new_cell_vertices,
    std::vector<std::int64_t>& new_global_cell_indices,
    std::vector<int>& new_cell_partition,
    std::map<std::int32_t, std::set<std::uint32_t>>& shared_cells)
{
  // This function takes the partition computed by the partitioner
  // stored in MeshPartition mp. Some cells go to multiple
  // destinations. Each cell is transmitted to its final
  // destination(s) including its global index, and the cell owner
  // (for ghost cells this will be different from the destination)

  log::log(PROGRESS, "Distribute cells during distributed mesh construction");

  common::Timer timer("Distribute cells");

  const std::size_t mpi_size = MPI::size(mpi_comm);
  const std::size_t mpi_rank = MPI::rank(mpi_comm);

  // Global offset, used to build global cell index, if not given
  std::int64_t global_offset
      = MPI::global_offset(mpi_comm, cell_vertices.rows(), true);
  bool build_global_index = global_cell_indices.empty();

  new_cell_partition.clear();
  shared_cells.clear();

  // Get dimensions
  const std::size_t num_local_cells = cell_vertices.rows();
  const std::size_t num_cell_vertices = cell_vertices.cols();
  dolfin_assert(mp.size() == num_local_cells);

  // Send all cells to their destinations including their global
  // indices.  First element of vector is cell count of unghosted
  // cells, second element is count of ghost cells.
  std::vector<std::vector<std::size_t>> send_cell_vertices(
      mpi_size, std::vector<std::size_t>(2, 0));

  for (std::uint32_t i = 0; i != mp.size(); ++i)
  {
    std::uint32_t num_procs = mp.num_procs(i);

    const std::uint32_t* sharing_procs = mp.procs(i);
    for (std::uint32_t j = 0; j < num_procs; ++j)
    {
      // Create reference to destination vector
      std::vector<std::size_t>& send_cell_dest
          = send_cell_vertices[sharing_procs[j]];

      // Count of ghost cells, followed by ghost processes
      if (num_procs > 1)
      {
        send_cell_dest.push_back(num_procs);
        send_cell_dest.insert(send_cell_dest.end(), sharing_procs,
                              sharing_procs + num_procs);
      }
      else // unghosted
        send_cell_dest.push_back(0);

      // Global cell index
      if (build_global_index)
        send_cell_dest.push_back(global_offset + i);
      else
        send_cell_dest.push_back(global_cell_indices[i]);

      // Global vertex indices
      send_cell_dest.insert(send_cell_dest.end(), cell_vertices.row(i).data(),
                            cell_vertices.row(i).data() + num_cell_vertices);

      // First entry is the owner, so this counts as a 'local' cell
      // subsequent entries are 'remote ghosts'
      if (j == 0)
        send_cell_dest[0]++;
      else
        send_cell_dest[1]++;
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

  new_cell_vertices.resize(all_count, num_cell_vertices);
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
        new_cell_vertices(idx, j) = *tmp_it++;

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
void MeshPartitioning::compute_vertex_mapping(
    MPI_Comm mpi_comm, Eigen::Ref<const EigenRowArrayXXi64> cell_vertices,
    std::vector<std::int64_t>& vertex_indices,
    Eigen::Ref<EigenRowArrayXXi32> local_cell_vertices)
{
  std::map<std::int64_t, std::int32_t> vertex_global_to_local;
  vertex_indices.clear();
  vertex_global_to_local.clear();

  const std::int32_t num_cells = cell_vertices.rows();
  const std::int32_t num_cell_vertices = cell_vertices.cols();

  local_cell_vertices.resize(num_cells, num_cell_vertices);

  // Get set of unique vertices from cells. Remap cell_vertices to
  // local_cell_vertices, starting from 0. Record the global indices for
  // each local vertex in vertex_indices.

  std::int32_t v = 0;
  for (std::int32_t i = 0; i < num_cells; ++i)
  {
    for (std::int32_t j = 0; j < num_cell_vertices; ++j)
    {
      std::int64_t q = cell_vertices(i, j);
      auto map_it = vertex_global_to_local.insert({q, v});
      local_cell_vertices(i, j) = map_it.first->second;
      if (map_it.second)
      {
        vertex_indices.push_back(q);
        ++v;
      }
    }
  }
}
//-----------------------------------------------------------------------------
void MeshPartitioning::distribute_vertices(
    const MPI_Comm mpi_comm, Eigen::Ref<const EigenRowArrayXXd> points,
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
  const int gdim = points.cols();

  // Compute where (process number) the vertices we need are located
  std::vector<std::size_t> ranges(mpi_size);
  MPI::all_gather(mpi_comm, (std::size_t)points.rows(), ranges);
  for (std::uint32_t i = 1; i != ranges.size(); ++i)
    ranges[i] += ranges[i - 1];
  ranges.insert(ranges.begin(), 0);

  // Send global indices to the processes that own them, also recording
  // in local_indexing the original position on this process
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

  // Each remote process will put the requested point coordinates into a
  // block of memory on the local process.
  // Calculate offset position for each process, and attach to the sending data
  std::size_t offset = 0;
  for (int i = 0; i != mpi_size; ++i)
  {
    send_vertex_indices[i].push_back(offset);
    offset += (send_vertex_indices[i].size() - 1);
  }

  // Send required vertex indices to other processes, and receive
  // vertex indices required by other processes.
  std::vector<std::vector<std::size_t>> received_vertex_indices;
  MPI::all_to_all(mpi_comm, send_vertex_indices, received_vertex_indices);

  // Pop offsets off back of received data
  std::vector<std::size_t> remote_offsets;
  std::size_t num_received_indices = 0;
  for (auto& p : received_vertex_indices)
  {
    remote_offsets.push_back(p.back());
    p.pop_back();
    num_received_indices += p.size();
  }

  // Pop offset off back of sending arrays too, achieving
  // a clean transfer of the offset data from local to remote
  for (auto& p : send_vertex_indices)
    p.pop_back();

  // Array to receive data into with RMA
  // This is a block of memory which all remote processes can write into, by
  // using the offset (and size) transferred in previous all_to_all.
  EigenRowArrayXXd receive_coord_data(vertex_indices.size(), gdim);

  // Create local RMA window
  MPI_Win win;
  MPI_Win_create(receive_coord_data.data(),
                 sizeof(double) * vertex_indices.size() * gdim, sizeof(double),
                 MPI_INFO_NULL, mpi_comm, &win);
  MPI_Win_fence(0, win);

  // This memory block is to read from, and must remain in place until the
  // transfer is complete (after next MPI_Win_fence)
  EigenRowArrayXXd send_coord_data(num_received_indices, gdim);

  const std::pair<std::size_t, std::size_t> local_vertex_range
      = {ranges[mpi_rank], ranges[mpi_rank + 1]};
  // Convert global index to local index and put coordinate data in sending
  // array
  std::size_t local_index = 0;
  for (int p = 0; p < mpi_size; ++p)
  {
    const std::size_t local_index_0 = local_index;
    for (const auto& q : received_vertex_indices[p])
    {
      dolfin_assert(q >= local_vertex_range.first
                    && q < local_vertex_range.second);

      const std::size_t location = q - local_vertex_range.first;
      send_coord_data.row(local_index) = points.row(location);
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

  // Reorder coordinates according to local indexing
  local_index = 0;
  for (const auto& p : local_indexing)
  {
    for (const auto& v : p)
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
  for (const auto& p : received_vertex_indices)
    for (const auto& q : p)
    {
      dolfin_assert(q >= local_vertex_range.first
                    and q < local_vertex_range.second);
      const std::size_t local_index = q - local_vertex_range.first;
      ++n_sharing[local_index];
    }

  // Create an array of 'pointers' to shared entries (where number shared, p >
  // 1) Set to 0 for unshared entries Make space for two values: process number,
  // and local index on that process
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
      // Convert global to local index
      const std::size_t q = received_vertex_indices[p][i];
      const std::size_t local_index = q - local_vertex_range.first;
      if (n_sharing[local_index] > 0)
      {
        std::int32_t& location = offset[local_index];
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

  // Unpack and store to shared_vertices_local
  for (unsigned int p = 0; p < mpi_size; ++p)
  {
    const std::vector<std::uint32_t>& local_index_p = local_indexing[p];
    for (auto q = recv_sharing[p].begin(); q != recv_sharing[p].end();
         q += (*q + 2))
    {
      const std::size_t num_sharing = *q;
      const std::uint32_t local_index = local_index_p[*(q + 1)];
      std::set<std::uint32_t> sharing_processes(q + 2, q + 2 + num_sharing);

      auto it = shared_vertices_local.insert({local_index, sharing_processes});
      dolfin_assert(it.second);
    }
  }
}
//-----------------------------------------------------------------------------
