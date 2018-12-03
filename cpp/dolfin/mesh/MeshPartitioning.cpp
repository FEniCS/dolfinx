// Copyright (C) 2008-2014 Niclas Jansson, Ola Skavhaug, Anders Logg,
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "MeshPartitioning.h"
#include "CellType.h"
#include "DistributedMeshTools.h"
#include "Facet.h"
#include "Mesh.h"
#include "MeshEntity.h"
#include "MeshFunction.h"
#include "MeshTopology.h"
#include "MeshValueCollection.h"
#include "Vertex.h"
#include <algorithm>
#include <cstdint>
#include <dolfin/common/MPI.h>
#include <dolfin/common/Timer.h>
#include <dolfin/geometry/Point.h>
#include <dolfin/graph/CSRGraph.h>
#include <dolfin/graph/GraphBuilder.h>
#include <dolfin/graph/ParMETIS.h>
#include <dolfin/graph/SCOTCH.h>
#include <dolfin/log/log.h>
#include <iterator>
#include <map>
#include <memory>
#include <numeric>
#include <set>

using namespace dolfin;
using namespace dolfin::mesh;

//-----------------------------------------------------------------------------
mesh::Mesh MeshPartitioning::build_distributed_mesh(
    const MPI_Comm& comm, mesh::CellType::Type type,
    const Eigen::Ref<const EigenRowArrayXXd>& points,
    const Eigen::Ref<const EigenRowArrayXXi64>& cells,
    const std::vector<std::int64_t>& global_cell_indices,
    const mesh::GhostMode ghost_mode, std::string graph_partitioner)
{
  // Compute the cell partition
  PartitionData mp = partition_cells(comm, type, cells, graph_partitioner);

  // Check that we have some ghost information.
  int all_ghosts = MPI::sum(comm, mp.num_ghosts());
  if (all_ghosts == 0 && ghost_mode != mesh::GhostMode::none)
  {
    log::dolfin_error("MeshPartitioning.cpp", "build ghost mesh",
                      "Ghost cell information not available");
  }

  // Build mesh from local mesh data and provided cell partition
  mesh::Mesh mesh
      = build(comm, type, cells, points, global_cell_indices, ghost_mode, mp);

  // Initialise number of globally connected cells to each facet. This
  // is necessary to distinguish between facets on an exterior
  // boundary and facets on a partition boundary (see
  // https://bugs.launchpad.net/dolfin/+bug/733834).

  DistributedMeshTools::init_facet_cell_connections(mesh);

  return mesh;
}
//-----------------------------------------------------------------------------
PartitionData MeshPartitioning::partition_cells(
    const MPI_Comm& mpi_comm, mesh::CellType::Type type,
    const Eigen::Ref<const EigenRowArrayXXi64>& cell_vertices,
    const std::string partitioner)
{
  log::log(PROGRESS, "Compute partition of cells across processes");

  std::unique_ptr<mesh::CellType> cell_type(mesh::CellType::create(type));
  assert(cell_type);

  // Compute dual graph (for this partition)
  std::vector<std::vector<std::size_t>> local_graph;
  std::tuple<std::int32_t, std::int32_t, std::int32_t> graph_info;
  std::tie(local_graph, graph_info) = graph::GraphBuilder::compute_dual_graph(
      mpi_comm, cell_vertices, *cell_type);

  // Compute cell partition using partitioner from parameter system
  if (partitioner == "SCOTCH")
  {
    graph::CSRGraph<SCOTCH_Num> csr_graph(mpi_comm, local_graph);
    std::vector<std::size_t> weights;
    const std::int32_t num_ghost_nodes = std::get<0>(graph_info);
    return PartitionData(graph::SCOTCH::partition(mpi_comm, csr_graph, weights,
                                                  num_ghost_nodes));
  }
  else if (partitioner == "ParMETIS")
  {
#ifdef HAS_PARMETIS
    graph::CSRGraph<idx_t> csr_graph(mpi_comm, local_graph);
    return PartitionData(graph::ParMETIS::partition(mpi_comm, csr_graph));
#else
    throw std::runtime_error("ParMETIS not available");
#endif
  }
  else
    throw std::runtime_error("Unknown graph partitioner");

  return PartitionData({}, {});
}
//-----------------------------------------------------------------------------
mesh::Mesh MeshPartitioning::build(
    const MPI_Comm& comm, mesh::CellType::Type type,
    const Eigen::Ref<const EigenRowArrayXXi64>& cell_vertices,
    const Eigen::Ref<const EigenRowArrayXXd>& points,
    const std::vector<std::int64_t>& global_cell_indices,
    const mesh::GhostMode ghost_mode, const PartitionData& mp)
{
  // Distribute cells
  log::log(PROGRESS, "Distribute mesh cells");

  common::Timer timer("Distribute mesh cells");

  // Create CellType objects based on current cell type
  std::unique_ptr<mesh::CellType> cell_type(mesh::CellType::create(type));
  assert(cell_type);

  // Topological dimension
  const int tdim = cell_type->dim();

  // Send cells to owning process according to mp cell partition, and
  // receive cells that belong to this process. Also compute auxiliary
  // data related to sharing.
  EigenRowArrayXXi64 new_cell_vertices;
  std::vector<std::int64_t> new_global_cell_indices;
  std::vector<int> new_cell_partition;
  std::map<std::int32_t, std::set<std::uint32_t>> shared_cells;
  std::int32_t num_regular_cells;
  std::tie(new_cell_vertices, new_global_cell_indices, new_cell_partition,
           shared_cells, num_regular_cells)
      = distribute_cells(comm, cell_vertices, global_cell_indices, mp);

  if (ghost_mode == mesh::GhostMode::shared_vertex)
  {
    // Send/receive additional cells defined by connectivity to the shared
    // vertices.
    distribute_cell_layer(comm, num_regular_cells, shared_cells,
                          new_cell_vertices, new_global_cell_indices,
                          new_cell_partition);
  }
  else if (ghost_mode == mesh::GhostMode::none)
  {
    // Resize to remove all ghost cells
    new_cell_partition.resize(num_regular_cells);
    new_global_cell_indices.resize(num_regular_cells);
    new_cell_vertices.conservativeResize(num_regular_cells, Eigen::NoChange);
    shared_cells.clear();
  }

#ifdef HAS_SCOTCH
  // if (parameter::parameters["reorder_cells_gps"])
  // {
  //   // Allocate objects to hold re-ordering
  //   std::map<std::int32_t, std::set<std::uint32_t>> reordered_shared_cells;
  //   EigenRowArrayXXi64 reordered_cell_vertices;
  //   std::vector<std::int64_t> reordered_global_cell_indices;

  //   // Re-order cells
  //   std::tie(reordered_shared_cells, reordered_cell_vertices,
  //            reordered_global_cell_indices)
  //       = reorder_cells_gps(comm, num_regular_cells, *cell_type,
  //       shared_cells,
  //                           new_cell_vertices, new_global_cell_indices);

  //   // Update to re-ordered indices
  //   std::swap(shared_cells, reordered_shared_cells);
  //   new_cell_vertices = reordered_cell_vertices;
  //   std::swap(new_global_cell_indices, reordered_global_cell_indices);
  // }
#endif

  timer.stop();

  // Build mesh from points and distributed cells
  const std::int32_t num_ghosts = new_cell_vertices.rows() - num_regular_cells;
  mesh::Mesh mesh(comm, type, points, new_cell_vertices,
                  new_global_cell_indices, ghost_mode, num_ghosts);

  if (ghost_mode == mesh::GhostMode::none)
    return mesh;

  // Copy cell ownership (only needed for ghost cells)
  std::vector<std::uint32_t>& cell_owner = mesh.topology().cell_owner();
  cell_owner.clear();
  cell_owner.insert(cell_owner.begin(),
                    new_cell_partition.begin() + num_regular_cells,
                    new_cell_partition.end());

  // Assign map of shared cells (only needed for ghost cells)
  mesh.topology().shared_entities(tdim) = shared_cells;

  return mesh;
}
//-----------------------------------------------------------------------------
std::tuple<std::map<std::int32_t, std::set<std::uint32_t>>, EigenRowArrayXXi64,
           std::vector<std::int64_t>>
MeshPartitioning::reorder_cells_gps(
    MPI_Comm mpi_comm, const std::uint32_t num_regular_cells,
    const CellType& cell_type,
    const std::map<std::int32_t, std::set<std::uint32_t>>& shared_cells,
    const Eigen::Ref<const EigenRowArrayXXi64>& global_cell_vertices,
    const std::vector<std::int64_t>& global_cell_indices)
{
  std::cout
      << "WARNING: this function is probably broken. It needs careful testing."
      << std::endl;

  log::log(PROGRESS, "Re-order cells during distributed mesh construction");

  common::Timer timer("Reorder cells using GPS ordering");

  // FIXME: Should not use Graph private methods
  // Make dual graph from vertex indices, using GraphBuilder
  // FIXME: this should be reused later to add the facet-cell topology
  std::vector<std::vector<std::size_t>> local_graph;
  dolfin::graph::GraphBuilder::FacetCellMap facet_cell_map;
  std::tie(local_graph, facet_cell_map, std::ignore)
      = dolfin::graph::GraphBuilder::compute_local_dual_graph(
          mpi_comm, global_cell_vertices, cell_type);

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
      assert(*q >= local_cell_offset);
      const int local_index = *q - local_cell_offset;

      // Ignore ghost cells in connectivity
      if (local_index < (int)num_regular_cells)
        conn_set.insert(local_index);
    }
    g_dual.push_back(conn_set);
  }
  std::vector<int> remap;
  std::tie(remap, std::ignore) = dolfin::graph::SCOTCH::compute_gps(g_dual);

  // Add direct mapping for any ghost cells (not reordered)
  for (unsigned int j = remap.size(); j < global_cell_indices.size(); ++j)
    remap.push_back(j);

  EigenRowArrayXXi64 reordered_cell_vertices(global_cell_vertices.rows(),
                                             global_cell_vertices.cols());
  std::vector<std::int64_t> reordered_global_cell_indices(g_dual.size());
  for (std::uint32_t i = 0; i != g_dual.size(); ++i)
  {
    // Remap data
    const std::uint32_t j = remap[i];
    reordered_cell_vertices.row(j) = global_cell_vertices.row(i);
    reordered_global_cell_indices[j] = global_cell_indices[i];
  }

  std::map<std::int32_t, std::set<std::uint32_t>> reordered_shared_cells;
  for (auto p = shared_cells.begin(); p != shared_cells.end(); ++p)
  {
    const std::uint32_t cell_index = p->first;
    if (cell_index < num_regular_cells)
      reordered_shared_cells.insert({remap[cell_index], p->second});
    else
      reordered_shared_cells.insert(*p);
  }

  return std::make_tuple(std::move(reordered_shared_cells),
                         std::move(reordered_cell_vertices),
                         std::move(reordered_global_cell_indices));
}
//-----------------------------------------------------------------------------
void MeshPartitioning::distribute_cell_layer(
    MPI_Comm mpi_comm, const int num_regular_cells,
    std::map<std::int32_t, std::set<std::uint32_t>>& shared_cells,
    EigenRowArrayXXi64& cell_vertices,
    std::vector<std::int64_t>& global_cell_indices,
    std::vector<int>& cell_partition)
{
  common::Timer timer("Distribute cell layer");

  const int mpi_size = MPI::size(mpi_comm);
  const int mpi_rank = MPI::rank(mpi_comm);

  // Map from shared vertex to the set of cells containing it
  std::map<std::int64_t, std::vector<std::int64_t>> sh_vert_to_cell;

  // Global to local mapping of cell indices
  std::map<std::int64_t, int> cell_global_to_local;

  // Iterate only over ghost cells
  for (Eigen::Index i = num_regular_cells; i < cell_vertices.rows(); ++i)
  {
    // Add map entry for each vertex of ghost cells
    for (Eigen::Index p = 0; p < cell_vertices.cols(); ++p)
    {
      sh_vert_to_cell.insert(
          {cell_vertices(i, p), std::vector<std::int64_t>()});
    }

    cell_global_to_local.insert({global_cell_indices[i], i});
  }

  // Iterate only over regular (non-ghost) cells
  for (int i = 0; i < num_regular_cells; ++i)
  {
    for (Eigen::Index j = 0; j != cell_vertices.cols(); ++j)
    {
      // If vertex already in map, append local cell index to set
      auto vc_it = sh_vert_to_cell.find(cell_vertices(i, j));
      if (vc_it != sh_vert_to_cell.end())
      {
        cell_global_to_local.insert({global_cell_indices[i], i});
        vc_it->second.push_back(i);
      }
    }
  }

  // sh_vert_to_cell now contains a mapping from the vertices of
  // ghost cells to any regular cells which they are also incident with.

  // Send lists of cells/owners to "index owner" of vertex,
  // collating and sending back out...
  std::vector<std::vector<std::int64_t>> send_vertcells(mpi_size);
  std::vector<std::vector<std::int64_t>> recv_vertcells(mpi_size);
  for (const auto& vc_it : sh_vert_to_cell)
  {
    // Generate unique destination "index owner" based on vertex index
    const int dest = (vc_it.first) % mpi_size;

    std::vector<std::int64_t>& sendv = send_vertcells[dest];

    // Pack as [cell_global_index, this_vertex, [other_vertices]]
    for (const auto& q : vc_it.second)
    {
      sendv.push_back(global_cell_indices[q]);
      sendv.push_back(vc_it.first);
      for (Eigen::Index v = 0; v < cell_vertices.cols(); ++v)
      {
        if (cell_vertices(q, v) != vc_it.first)
          sendv.push_back(cell_vertices(q, v));
      }
    }
  }

  MPI::all_to_all(mpi_comm, send_vertcells, recv_vertcells);

  const std::uint32_t num_cell_vertices = cell_vertices.cols();

  // Collect up cells on common vertices

  // Reset map
  sh_vert_to_cell.clear();
  std::vector<std::int64_t> cell_set;
  for (int i = 0; i < mpi_size; ++i)
  {
    const std::vector<std::int64_t>& recv_i = recv_vertcells[i];
    for (auto q = recv_i.begin(); q != recv_i.end(); q += num_cell_vertices + 1)
    {
      const std::size_t vertex_index = *(q + 1);
      // Packing: [owner, cell_index, this_vertex, [other_vertices]]
      cell_set = {i};
      cell_set.insert(cell_set.end(), q, q + num_cell_vertices + 1);

      // Look for vertex in map, and add the attached cell
      auto it = sh_vert_to_cell.insert({vertex_index, cell_set});
      if (!it.second)
        it.first->second.insert(it.first->second.end(), cell_set.begin(),
                                cell_set.end());
    }
  }

  // Clear sending arrays
  send_vertcells = std::vector<std::vector<std::int64_t>>(mpi_size);

  // Send back out to all processes which share the same vertex
  for (const auto& p : sh_vert_to_cell)
  {
    for (auto q = p.second.begin(); q != p.second.end();
         q += (num_cell_vertices + 2))
    {
      send_vertcells[*q].insert(send_vertcells[*q].end(), p.second.begin(),
                                p.second.end());
    }
  }

  MPI::all_to_all(mpi_comm, send_vertcells, recv_vertcells);

  // Count up new cells, assign local index, set owner
  // and initialise shared_cells

  const std::uint32_t num_cells = cell_vertices.rows();
  std::uint32_t count = num_cells;

  for (const auto& p : recv_vertcells)
  {
    for (auto q = p.begin(); q != p.end(); q += num_cell_vertices + 2)
    {
      const std::int64_t owner = *q;
      const std::int64_t cell_index = *(q + 1);

      auto cell_insert = cell_global_to_local.insert({cell_index, count});
      if (cell_insert.second)
      {
        shared_cells.insert({count, std::set<std::uint32_t>()});
        global_cell_indices.push_back(cell_index);
        cell_partition.push_back(owner);
        ++count;
      }
    }
  }

  // Add received cells and update sharing information for cells
  cell_vertices.conservativeResize(count, num_cell_vertices);

  // Set of processes and cells sharing the same vertex
  std::set<std::uint32_t> sharing_procs;
  std::vector<std::size_t> sharing_cells;

  std::size_t last_vertex = std::numeric_limits<std::size_t>::max();
  for (const auto& p : recv_vertcells)
  {
    for (auto q = p.begin(); q != p.end(); q += num_cell_vertices + 2)
    {
      const int owner = *q;
      const std::size_t cell_index = *(q + 1);
      const std::size_t shared_vertex = *(q + 2);
      const std::size_t local_index
          = cell_global_to_local.find(cell_index)->second;

      // Add vertices to new cells
      if (local_index >= num_cells)
      {
        for (std::uint32_t j = 0; j != num_cell_vertices; ++j)
          cell_vertices(local_index, j) = *(q + j + 2);
      }

      // If starting on a new shared vertex, dump old data into
      // shared_cells
      if (shared_vertex != last_vertex)
      {
        last_vertex = shared_vertex;
        for (const auto& c : sharing_cells)
        {
          auto it = shared_cells.insert({c, sharing_procs});
          if (!it.second)
            it.first->second.insert(sharing_procs.begin(), sharing_procs.end());
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

  // Dump data from final vertex into shared_cells
  for (const auto& c : sharing_cells)
  {
    auto it = shared_cells.insert({c, sharing_procs});
    if (!it.second)
      it.first->second.insert(sharing_procs.begin(), sharing_procs.end());
  }
}
//-----------------------------------------------------------------------------
std::tuple<EigenRowArrayXXi64, std::vector<std::int64_t>, std::vector<int>,
           std::map<std::int32_t, std::set<std::uint32_t>>, std::int32_t>
MeshPartitioning::distribute_cells(
    const MPI_Comm mpi_comm,
    const Eigen::Ref<const EigenRowArrayXXi64>& cell_vertices,
    const std::vector<std::int64_t>& global_cell_indices,
    const PartitionData& mp)
{
  // This function takes the partition computed by the partitioner
  // stored in PartitionData mp. Some cells go to multiple
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

  // Get dimensions
  const std::size_t num_local_cells = cell_vertices.rows();
  const std::size_t num_cell_vertices = cell_vertices.cols();
  assert(mp.size() == num_local_cells);

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

  EigenRowArrayXXi64 new_cell_vertices(all_count, num_cell_vertices);
  std::vector<std::int64_t> new_global_cell_indices(all_count);
  std::vector<int> new_cell_partition(all_count);

  // Unpack received data
  // Create a map from cells which are shared, to the remote processes
  // which share them - corral ghost cells to end of range
  std::size_t c = 0;
  std::size_t gc = local_count;
  std::map<std::int32_t, std::set<std::uint32_t>> shared_cells;
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

      assert(idx < new_cell_partition.size());
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

  assert(c == local_count);
  assert(gc == all_count);

  return std::make_tuple(
      std::move(new_cell_vertices), std::move(new_global_cell_indices),
      std::move(new_cell_partition), std::move(shared_cells), local_count);
}
//-----------------------------------------------------------------------------
std::tuple<std::uint64_t, std::vector<std::int64_t>, EigenRowArrayXXi32>
MeshPartitioning::compute_point_mapping(
    std::uint32_t num_cell_vertices,
    const Eigen::Ref<const EigenRowArrayXXi64>& cell_points,
    const std::vector<std::uint8_t>& cell_permutation)
{
  const std::uint32_t num_cells = cell_points.rows();
  const std::uint32_t num_cell_points = cell_points.cols();

  // Cell points in local indexing
  EigenRowArrayXXi32 local_cell_points(num_cells, num_cell_points);

  // Local-to-global map for points
  std::vector<std::int64_t> point_local_to_global;

  // Get set of unique points from cells. Remap cell_points to
  // local_cell_points, starting from 0. Record the global indices for
  // each local point in point_indices.

  // Loop over cells mapping vertices (but not other points)
  std::int32_t nv = 0;
  std::map<std::int64_t, std::int32_t> point_global_to_local;
  for (std::uint32_t c = 0; c < num_cells; ++c)
  {
    // Loop over cell points
    for (std::uint32_t v = 0; v < num_cell_vertices; ++v)
    {
      // Get global cell index
      std::int64_t q = cell_points(c, v);

      // Insert (global_vertex_index, local_vertex_index) into map
      auto map_it = point_global_to_local.insert({q, nv});

      // Set local index in cell vertex list
      local_cell_points(c, cell_permutation[v]) = map_it.first->second;

      // If global index seen for first time, add to local-to-global map
      if (map_it.second)
      {
        point_local_to_global.push_back(q);
        ++nv;
      }
    }
  }

  const std::uint32_t num_local_vertices = nv;

  // Repeat and map non-vertex points
  for (std::uint32_t c = 0; c < num_cells; ++c)
  {
    // Loop over cell points
    for (std::uint32_t v = num_cell_vertices; v < num_cell_points; ++v)
    {
      // Get global cell index
      std::int64_t q = cell_points(c, v);

      // Insert (global_vertex_index, local_vertex_index) into map
      auto map_it = point_global_to_local.insert({q, nv});

      // Set local index in cell vertex list
      local_cell_points(c, cell_permutation[v]) = map_it.first->second;

      // If global index seen for first time, add to local-to-global map
      if (map_it.second)
      {
        point_local_to_global.push_back(q);
        ++nv;
      }
    }
  }

  return std::make_tuple(num_local_vertices, std::move(point_local_to_global),
                         std::move(local_cell_points));
}
//-----------------------------------------------------------------------------
std::pair<EigenRowArrayXXd, std::map<std::int32_t, std::set<std::uint32_t>>>
MeshPartitioning::distribute_points(
    const MPI_Comm mpi_comm, const Eigen::Ref<const EigenRowArrayXXd>& points,
    const std::vector<std::int64_t>& global_point_indices)
{
  // This function distributes all points (coordinates and
  // local-to-global mapping) according to the cells that are stored
  // on each process. This happens in several stages: First each
  // process figures out which points it needs (by looking at its
  // cells) and where those points are located. That information is
  // then distributed so that each process learns where it needs to
  // send its points.

  // Create data structures that will be returned
  EigenRowArrayXXd point_coordinates(global_point_indices.size(),
                                     points.cols());

  log::log(PROGRESS, "Distribute points during distributed mesh construction");
  common::Timer timer("Distribute points");

  // Get number of processes
  const int mpi_size = MPI::size(mpi_comm);
  const int mpi_rank = MPI::rank(mpi_comm);

  // Get geometric dimension
  const int gdim = points.cols();

  // Compute where (process number) the points we need are located
  std::vector<std::size_t> ranges(mpi_size);
  MPI::all_gather(mpi_comm, (std::size_t)points.rows(), ranges);
  for (std::uint32_t i = 1; i != ranges.size(); ++i)
    ranges[i] += ranges[i - 1];
  ranges.insert(ranges.begin(), 0);

  // Send global indices to the processes that own them, also recording
  // in local_indexing the original position on this process
  std::vector<std::vector<std::size_t>> send_point_indices(mpi_size);
  std::vector<std::vector<std::uint32_t>> local_indexing(mpi_size);
  for (unsigned int i = 0; i != global_point_indices.size(); ++i)
  {
    const std::size_t required_point = global_point_indices[i];
    const int location
        = std::upper_bound(ranges.begin(), ranges.end(), required_point)
          - ranges.begin() - 1;
    send_point_indices[location].push_back(required_point);
    local_indexing[location].push_back(i);
  }

  // Each remote process will put the requested point coordinates into a
  // block of memory on the local process.
  // Calculate offset position for each process, and attach to the sending
  // data
  std::size_t offset = 0;
  for (int i = 0; i != mpi_size; ++i)
  {
    send_point_indices[i].push_back(offset);
    offset += (send_point_indices[i].size() - 1);
  }

  // Send required point indices to other processes, and receive
  // point indices required by other processes.
  std::vector<std::vector<std::size_t>> received_point_indices;
  MPI::all_to_all(mpi_comm, send_point_indices, received_point_indices);

  // Pop offsets off back of received data
  std::vector<std::size_t> remote_offsets;
  std::size_t num_received_indices = 0;
  for (auto& p : received_point_indices)
  {
    remote_offsets.push_back(p.back());
    p.pop_back();
    num_received_indices += p.size();
  }

  // Pop offset off back of sending arrays too, achieving
  // a clean transfer of the offset data from local to remote
  for (auto& p : send_point_indices)
    p.pop_back();

  // Array to receive data into with RMA
  // This is a block of memory which all remote processes can write into, by
  // using the offset (and size) transferred in previous all_to_all.
  EigenRowArrayXXd receive_coord_data(global_point_indices.size(), gdim);

  // Create local RMA window
  MPI_Win win;
  MPI_Win_create(receive_coord_data.data(),
                 sizeof(double) * global_point_indices.size() * gdim,
                 sizeof(double), MPI_INFO_NULL, mpi_comm, &win);
  MPI_Win_fence(0, win);

  // This memory block is to read from, and must remain in place until the
  // transfer is complete (after next MPI_Win_fence)
  EigenRowArrayXXd send_coord_data(num_received_indices, gdim);

  const std::pair<std::size_t, std::size_t> local_point_range
      = {ranges[mpi_rank], ranges[mpi_rank + 1]};
  // Convert global index to local index and put coordinate data in sending
  // array
  std::size_t local_index = 0;
  for (int p = 0; p < mpi_size; ++p)
  {
    if (received_point_indices[p].size() > 0)
    {
      const std::size_t local_index_0 = local_index;
      for (const auto& q : received_point_indices[p])
      {
        assert(q >= local_point_range.first && q < local_point_range.second);

        const std::size_t location = q - local_point_range.first;
        send_coord_data.row(local_index) = points.row(location);
        ++local_index;
      }

      const std::size_t local_size = (local_index - local_index_0) * gdim;
      MPI_Put(send_coord_data.data() + local_index_0 * gdim, local_size,
              MPI_DOUBLE, p, remote_offsets[p] * gdim, local_size, MPI_DOUBLE,
              win);
    }
  }

  // Meanwhile, redistribute received_point_indices as point sharing
  // information
  const std::map<std::int32_t, std::set<std::uint32_t>> shared_points_local
      = build_shared_points(mpi_comm, received_point_indices, local_point_range,
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
      point_coordinates.row(v) = receive_coord_data.row(local_index);
      ++local_index;
    }
  }

  return {std::move(point_coordinates), std::move(shared_points_local)};
}
//-----------------------------------------------------------------------------
std::map<std::int32_t, std::set<std::uint32_t>>
MeshPartitioning::build_shared_points(
    MPI_Comm mpi_comm,
    const std::vector<std::vector<std::size_t>>& received_point_indices,
    const std::pair<std::size_t, std::size_t> local_point_range,
    const std::vector<std::vector<std::uint32_t>>& local_indexing)
{
  log::log(PROGRESS,
           "Build shared points during distributed mesh construction");

  const std::uint32_t mpi_size = MPI::size(mpi_comm);

  // Count number sharing each local point
  std::vector<std::int32_t> n_sharing(
      local_point_range.second - local_point_range.first, 0);
  for (const auto& p : received_point_indices)
    for (const auto& q : p)
    {
      assert(q >= local_point_range.first and q < local_point_range.second);
      const std::size_t local_index = q - local_point_range.first;
      ++n_sharing[local_index];
    }

  // Create an array of 'pointers' to shared entries
  // (where number shared, p > 1).
  // Set to 0 for unshared entries. Make space for two values: process
  // number, and local index on that process
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
  // received_point_indices to send back to originating process
  std::vector<std::int32_t> process_list(index);
  for (std::uint32_t p = 0; p < mpi_size; ++p)
    for (unsigned int i = 0; i < received_point_indices[p].size(); ++i)
    {
      // Convert global to local index
      const std::size_t q = received_point_indices[p][i];
      const std::size_t local_index = q - local_point_range.first;
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

  // Unpack and store to shared_points_local
  std::map<std::int32_t, std::set<std::uint32_t>> shared_points_local;
  for (unsigned int p = 0; p < mpi_size; ++p)
  {
    const std::vector<std::uint32_t>& local_index_p = local_indexing[p];
    for (auto q = recv_sharing[p].begin(); q != recv_sharing[p].end();
         q += (*q + 2))
    {
      const std::size_t num_sharing = *q;
      const std::uint32_t local_index = local_index_p[*(q + 1)];
      std::set<std::uint32_t> sharing_processes(q + 2, q + 2 + num_sharing);

      auto it = shared_points_local.insert({local_index, sharing_processes});
      assert(it.second);
    }
  }

  return shared_points_local;
}
//-----------------------------------------------------------------------------
std::pair<std::int64_t, std::vector<std::int64_t>>
MeshPartitioning::build_global_vertex_indices(
    MPI_Comm mpi_comm, std::uint32_t num_vertices,
    const std::vector<std::int64_t>& global_point_indices,
    const std::map<std::int32_t, std::set<std::uint32_t>>& shared_points)
{
  // Find out how many vertices are locally 'owned' and number them
  std::vector<std::int64_t> global_vertex_indices(num_vertices);
  const std::uint32_t mpi_rank = MPI::rank(mpi_comm);
  const std::uint32_t mpi_size = MPI::size(mpi_comm);
  std::vector<std::vector<std::int64_t>> send_data(mpi_size);
  std::vector<std::int64_t> recv_data(mpi_size);

  std::int64_t v = 0;
  for (std::uint32_t i = 0; i < num_vertices; ++i)
  {
    const auto it = shared_points.find(i);
    if (it == shared_points.end())
    {
      // local
      global_vertex_indices[i] = v;
      ++v;
    }
    else
    {
      // Owned locally if rank less than first entry
      if (mpi_rank < *it->second.begin())
      {
        global_vertex_indices[i] = v;
        for (auto p : it->second)
        {
          send_data[p].push_back(global_point_indices[i]);
          send_data[p].push_back(v);
        }
        ++v;
      }
    }
  }

  // Now have numbered all vertices locally so can get global
  // size and local offset
  std::int64_t num_vertices_global = MPI::sum(mpi_comm, v);
  std::int64_t offset = MPI::global_offset(mpi_comm, v, true);

  // Add offset to send_data
  for (auto& p : send_data)
    for (auto q = p.begin(); q != p.end(); q += 2)
      *(q + 1) += offset;

  // Receive indices of vertices owned elsewhere into map
  MPI::all_to_all(mpi_comm, send_data, recv_data);
  std::map<std::int64_t, std::int64_t> global_point_to_vertex;
  for (auto p = recv_data.begin(); p != recv_data.end(); p += 2)
  {
    auto it = global_point_to_vertex.insert({*p, *(p + 1)});
    assert(it.second);
  }

  // Adjust global_vertex_indices either by adding offset
  // or inserting remote index
  for (std::uint32_t i = 0; i < num_vertices; ++i)
  {
    const auto it = global_point_to_vertex.find(global_point_indices[i]);
    if (it == global_point_to_vertex.end())
      global_vertex_indices[i] += offset;
    else
      global_vertex_indices[i] = it->second;
  }

  return {num_vertices_global, std::move(global_vertex_indices)};
}
//-----------------------------------------------------------------------------
