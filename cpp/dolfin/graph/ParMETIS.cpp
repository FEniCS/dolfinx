// Copyright (C) 2008-2016 Niclas Jansson, Ola Skavhaug, Anders Logg,
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "ParMETIS.h"
#include "CSRGraph.h"
#include "Graph.h"
#include "GraphBuilder.h"
#include <dolfin/common/MPI.h>
#include <dolfin/common/Timer.h>

#ifdef HAS_PARMETIS
#include <parmetis.h>
#endif

using namespace dolfin;

#ifdef HAS_PARMETIS

// namespace
// {
// // Create a dual graph from the cell-vertex topology using ParMETIS
// // built in ParMETIS_V3_Mesh2Dual
// dolfin::graph::CSRGraph<idx_t>
// build_dual_graph(MPI_Comm mpi_comm,
//                  Eigen::Ref<const EigenRowArrayXXi64> cell_vertices,
//                  const int num_vertices_per_cell)
// {
//   common::Timer timer("Build mesh dual graph (ParMETIS)");

//   // ParMETIS data structures
//   std::vector<idx_t> elmdist, eptr, eind;

//   // Get number of processes and process number
//   const std::int32_t num_processes = dolfin::MPI::size(mpi_comm);

//   // Get dimensions of local mesh_data
//   const std::int32_t num_local_cells = cell_vertices.rows();
//   const std::int32_t num_cell_vertices = num_vertices_per_cell;
//   assert((std::int32_t)cell_vertices.rows() == num_cell_vertices);

//   // Check that number of local graph nodes (cells) is > 0
//   if (num_local_cells == 0)
//   {
//     throw std::runtime_error(
//         "ParMETIS cannot be used if a process has no cells (graph nodes). Use
//         " "SCOTCH to perform partitioning instead");
//   }

//   // Communicate number of cells on each process between all processors
//   std::vector<std::int32_t> num_cells;
//   dolfin::MPI::all_gather(mpi_comm, num_local_cells, num_cells);

//   // Build elmdist array with cell offsets for all processors
//   elmdist.assign(num_processes + 1, 0);
//   for (std::int32_t i = 1; i < num_processes + 1; ++i)
//     elmdist[i] = elmdist[i - 1] + num_cells[i - 1];

//   eptr.resize(num_local_cells + 1);
//   eind.assign(num_local_cells * num_cell_vertices, 0);
//   for (std::int32_t i = 0; i < num_local_cells; i++)
//   {
//     eptr[i] = i * num_cell_vertices;
//     for (std::int32_t j = 0; j < num_cell_vertices; j++)
//       eind[eptr[i] + j] = cell_vertices(i, j);
//   }
//   eptr[num_local_cells] = num_local_cells * num_cell_vertices;

//   assert(!eptr.empty());
//   assert(!eind.empty());

//   // Number of nodes shared for dual graph (partition along facets)
//   idx_t ncommonnodes = num_cell_vertices - 1;

//   assert(!eptr.empty());
//   assert(!eind.empty());

//   // Could use GraphBuilder::compute_dual_graph() instead
//   common::Timer timer1("ParMETIS: call ParMETIS_V3_Mesh2Dual");
//   idx_t* xadj = nullptr;
//   idx_t* adjncy = nullptr;
//   idx_t numflag = 0;
//   int err = ParMETIS_V3_Mesh2Dual(elmdist.data(), eptr.data(), eind.data(),
//                                   &numflag, &ncommonnodes, &xadj, &adjncy,
//                                   &mpi_comm);
//   assert(err == METIS_OK);
//   timer1.stop();

//   // Build graph
//   dolfin::graph::CSRGraph<idx_t> csr_graph(mpi_comm, xadj, adjncy,
//                                            num_local_cells);

//   // Clean up ParMETIS
//   METIS_Free(xadj);
//   METIS_Free(adjncy);

//   return csr_graph;
// }
// } // namespace

//-----------------------------------------------------------------------------
std::pair<std::vector<int>, std::map<std::int64_t, std::vector<int>>>
dolfin::graph::ParMETIS::partition(MPI_Comm mpi_comm, idx_t nparts,
                                   const CSRGraph<idx_t>& csr_graph)
{
  std::map<std::int64_t, std::vector<int>> ghost_procs;

  common::Timer timer("Compute graph partition (ParMETIS)");

  // Options for ParMETIS
  idx_t options[3];
  options[0] = 1;
  options[1] = 0;
  options[2] = 15;

  // Strange weight arrays needed by ParMETIS
  idx_t ncon = 1;

  // Prepare remaining arguments for ParMETIS
  idx_t* elmwgt = nullptr;
  idx_t wgtflag = 0;
  idx_t edgecut = 0;
  idx_t numflag = 0;
  std::vector<real_t> tpwgts(ncon * nparts, 1.0 / static_cast<real_t>(nparts));
  std::vector<real_t> ubvec(ncon, 1.05);

  // Note: ParMETIS is not const-correct, so we throw away const-ness
  // and trust ParMETIS to not modify the data.

  // Call ParMETIS to partition graph
  common::Timer timer1("ParMETIS: call ParMETIS_V3_PartKway");
  const std::int32_t num_local_cells = csr_graph.size();
  std::vector<idx_t> part(num_local_cells);
  assert(!part.empty());
  int err = ParMETIS_V3_PartKway(
      const_cast<idx_t*>(csr_graph.node_distribution().data()),
      const_cast<idx_t*>(csr_graph.nodes().data()),
      const_cast<idx_t*>(csr_graph.edges().data()), elmwgt, nullptr, &wgtflag,
      &numflag, &ncon, &nparts, tpwgts.data(), ubvec.data(), options, &edgecut,
      part.data(), &mpi_comm);
  assert(err == METIS_OK);
  timer1.stop();

  common::Timer timer2("Compute graph halo data (ParMETIS)");

  // Work out halo cells for current division of dual graph
  const auto& elmdist = csr_graph.node_distribution();
  const auto& xadj = csr_graph.nodes();
  const auto& adjncy = csr_graph.edges();
  const std::int32_t num_processes = dolfin::MPI::size(mpi_comm);
  const std::int32_t process_number = dolfin::MPI::rank(mpi_comm);
  const idx_t elm_begin = elmdist[process_number];
  const idx_t elm_end = elmdist[process_number + 1];
  const std::int32_t ncells = elm_end - elm_begin;

  std::map<idx_t, std::set<std::int32_t>> halo_cell_to_remotes;
  // local indexing "i"
  for (int i = 0; i < ncells; i++)
  {
    for (auto other_cell :
         csr_graph[i]) // idx_t j = xadj[i]; j != xadj[i + 1]; ++j)
    {
      //      const idx_t other_cell = adjncy[j];
      if (other_cell < elm_begin || other_cell >= elm_end)
      {
        const int remote
            = std::upper_bound(elmdist.begin(), elmdist.end(), other_cell)
              - elmdist.begin() - 1;

        assert(remote < num_processes);
        if (halo_cell_to_remotes.find(i) == halo_cell_to_remotes.end())
          halo_cell_to_remotes[i] = std::set<std::int32_t>();
        halo_cell_to_remotes[i].insert(remote);
      }
    }
  }

  // Do halo exchange of cell partition data
  std::vector<std::vector<std::int64_t>> send_cell_partition(num_processes);
  std::vector<std::int64_t> recv_cell_partition;
  for (const auto& hcell : halo_cell_to_remotes)
  {
    for (auto proc : hcell.second)
    {
      assert(proc < num_processes);

      // global cell number
      send_cell_partition[proc].push_back(hcell.first + elm_begin);

      // partitioning
      send_cell_partition[proc].push_back(part[hcell.first]);
    }
  }

  // Actual halo exchange
  dolfin::MPI::all_to_all(mpi_comm, send_cell_partition, recv_cell_partition);

  // Construct a map from all currently foreign cells to their new
  // partition number
  std::map<std::int64_t, std::int32_t> cell_ownership;
  for (auto p = recv_cell_partition.begin(); p != recv_cell_partition.end();
       p += 2)
  {
    cell_ownership[*p] = *(p + 1);
  }

  // Generate mapping for where new boundary cells need to be sent
  for (std::int32_t i = 0; i < ncells; i++)
  {
    const std::size_t proc_this = part[i];
    for (idx_t j = xadj[i]; j < xadj[i + 1]; ++j)
    {
      const idx_t other_cell = adjncy[j];
      std::size_t proc_other;

      if (other_cell < elm_begin || other_cell >= elm_end)
      { // remote cell - should be in map
        const auto find_other_proc = cell_ownership.find(other_cell);
        assert(find_other_proc != cell_ownership.end());
        proc_other = find_other_proc->second;
      }
      else
        proc_other = part[other_cell - elm_begin];

      if (proc_this != proc_other)
      {
        auto map_it = ghost_procs.find(i);
        if (map_it == ghost_procs.end())
        {
          std::vector<std::int32_t> sharing_processes;
          sharing_processes.push_back(proc_this);
          sharing_processes.push_back(proc_other);
          ghost_procs.insert({i, sharing_processes});
        }
        else
        {
          // Add to vector if not already there
          auto it = std::find(map_it->second.begin(), map_it->second.end(),
                              proc_other);
          if (it == map_it->second.end())
            map_it->second.push_back(proc_other);
        }
      }
    }
  }

  timer2.stop();

  return std::make_pair(std::vector<int>(part.begin(), part.end()),
                        std::move(ghost_procs));
}
//-----------------------------------------------------------------------------
template <typename T>
std::vector<int> dolfin::graph::ParMETIS::adaptive_repartition(
    MPI_Comm mpi_comm, const CSRGraph<T>& csr_graph, double weight)
{
  common::Timer timer(
      "Compute graph partition (ParMETIS Adaptive Repartition)");

  // Options for ParMETIS
  idx_t options[4];
  options[0] = 1;
  options[1] = 0;
  options[2] = 15;
  options[3] = PARMETIS_PSR_UNCOUPLED;
  // For repartition, PARMETIS_PSR_COUPLED seems to suppress all
  // migration if already balanced.  Try PARMETIS_PSR_UNCOUPLED for
  // better edge cut.

  common::Timer timer1("ParMETIS: call ParMETIS_V3_AdaptiveRepart");
  real_t _itr = weight;
  std::vector<idx_t> part(csr_graph.size());
  std::vector<idx_t> vsize(part.size(), 1);
  assert(!part.empty());

  // Number of partitions (one for each process)
  idx_t nparts = dolfin::MPI::size(mpi_comm);

  // Remaining ParMETIS parameters
  idx_t ncon = 1;
  idx_t* elmwgt = nullptr;
  idx_t wgtflag = 0;
  idx_t edgecut = 0;
  idx_t numflag = 0;
  std::vector<real_t> tpwgts(ncon * nparts, 1.0 / static_cast<real_t>(nparts));
  std::vector<real_t> ubvec(ncon, 1.05);

  // Call ParMETIS to repartition graph
  int err = ParMETIS_V3_AdaptiveRepart(
      csr_graph.node_distribution().data(), csr_graph.nodes().data(),
      csr_graph.edges().data(), elmwgt, nullptr, vsize.data(), &wgtflag,
      &numflag, &ncon, &nparts, tpwgts.data(), ubvec.data(), &_itr, options,
      &edgecut, part.data(), &mpi_comm);
  assert(err == METIS_OK);
  timer1.stop();

  // Copy cell partition data and return
  return std::vector<int>(part.begin(), part.end());
}
//-----------------------------------------------------------------------------
template <typename T>
std::vector<int> dolfin::graph::ParMETIS::refine(MPI_Comm mpi_comm,
                                                 const CSRGraph<T>& csr_graph)
{
  common::Timer timer("Compute graph partition (ParMETIS Refine)");

  // Get some MPI data
  const std::int32_t process_number = dolfin::MPI::rank(mpi_comm);

  // Options for ParMETIS
  idx_t options[4];
  options[0] = 1;
  options[1] = 0;
  options[2] = 15;
  // options[3] = PARMETIS_PSR_UNCOUPLED;

  // For repartition, PARMETIS_PSR_COUPLED seems to suppress all
  // migration if already balanced.  Try PARMETIS_PSR_UNCOUPLED for
  // better edge cut.

  // Partitioning array to be computed by ParMETIS. Prefill with
  // process_number.
  const std::int32_t num_local_cells = csr_graph.size();
  std::vector<idx_t> part(num_local_cells, process_number);
  assert(!part.empty());

  // Number of partitions (one for each process)
  idx_t nparts = dolfin::MPI::size(mpi_comm);
  // Remaining ParMETIS parameters
  idx_t ncon = 1;
  idx_t* elmwgt = nullptr;
  idx_t wgtflag = 0;
  idx_t edgecut = 0;
  idx_t numflag = 0;
  std::vector<real_t> tpwgts(ncon * nparts, 1.0 / static_cast<real_t>(nparts));
  std::vector<real_t> ubvec(ncon, 1.05);

  // Call ParMETIS to partition graph
  common::Timer timer1("ParMETIS: call ParMETIS_V3_RefineKway");
  int err = ParMETIS_V3_RefineKway(
      csr_graph.node_distribution().data(), csr_graph.nodes().data(),
      csr_graph.edges().data(), elmwgt, nullptr, &wgtflag, &numflag, &ncon,
      &nparts, tpwgts.data(), ubvec.data(), options, &edgecut, part.data(),
      &mpi_comm);
  assert(err == METIS_OK);
  timer1.stop();

  // Copy cell partition data
  return std::vector<int>(part.begin(), part.end());
}
//-----------------------------------------------------------------------------
#endif
