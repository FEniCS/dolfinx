// Copyright (C) 2008-2011 Niclas Jansson, Ola Skavhaug and Anders Logg
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
// Modified by Garth N. Wells 2010
// Modified by Chris Richardson 2013
//
// First added:  2010-02-10
// Last changed: 2014-01-09

#include <dolfin/common/MPI.h>
#include <dolfin/common/Timer.h>
#include <dolfin/log/log.h>
#include <dolfin/mesh/CellType.h>
#include <dolfin/parameter/GlobalParameters.h>
#include "GraphBuilder.h"
#include "ParMETIS.h"

#ifdef HAS_PARMETIS
#include <parmetis.h>
#endif

using namespace dolfin;

#ifdef HAS_PARMETIS
//-----------------------------------------------------------------------------
void ParMETIS::compute_partition(const MPI_Comm mpi_comm,
                                 std::vector<int>& cell_partition,
                                 std::map<std::int64_t, std::vector<int>>& ghost_procs,
                                 const boost::multi_array<std::int64_t, 2>& cell_vertices,
                                 const std::size_t num_global_vertices,
                                 const CellType& cell_type,
                                 const std::string mode)
{
  // Duplicate MPI communicator (ParMETIS does not take const
  // arguments, so duplicate communicator to be sure it isn't changed)
  MPI_Comm comm;
  MPI_Comm_dup(mpi_comm, &comm);

  const std::int8_t tdim = cell_type.dim();
  const std::int8_t num_vertices_per_cell = cell_type.num_vertices(tdim);

  // Create data structures to hold graph
  CSRGraph<idx_t> csr_graph;

  // Use ParMETIS or DOLFIN dual graph
  bool use_parmetis_dual_graph = false;

  if (use_parmetis_dual_graph)
  {
    // Build dual graph using ParMETIS builder
    csr_graph = dual_graph(mpi_comm, cell_vertices, num_vertices_per_cell);
  }
  else
  {
    // Compute dual graph with DOLFIN
    std::vector<std::vector<std::size_t>> local_graph;
    std::set<std::int64_t> ghost_vertices;
    GraphBuilder::compute_dual_graph(mpi_comm, cell_vertices, cell_type,
                                     num_global_vertices, local_graph,
                                     ghost_vertices);

    csr_graph = CSRGraph<idx_t>(mpi_comm, local_graph);

  }

  // Partition graph
  if (mode == "partition")
    partition(comm, csr_graph, cell_partition, ghost_procs);
  else if (mode == "adaptive_repartition")
    adaptive_repartition(comm, csr_graph, cell_partition);
  else if (mode == "refine")
    refine(comm, csr_graph, cell_partition);
  else
  {
    dolfin_error("ParMETIS.cpp",
                 "compute mesh partitioning using ParMETIS",
                 "partition model %s is unknown. Must be \"partition\", \"adactive_partition\" or \"refine\"",
                 mode.c_str());
  }

  MPI_Comm_free(&comm);
}
//-----------------------------------------------------------------------------
template <typename T>
void ParMETIS::partition(MPI_Comm mpi_comm,
                         CSRGraph<T>& csr_graph,
                         std::vector<int>& cell_partition,
                         std::map<std::int64_t, std::vector<int>>& ghost_procs)
{
  Timer timer("Compute graph partition (ParMETIS)");

  // Options for ParMETIS
  idx_t options[3];
  options[0] = 1;
  options[1] = 0;
  options[2] = 15;

  // Number of partitions (one for each process)
  idx_t nparts = MPI::size(mpi_comm);

  // Strange weight arrays needed by ParMETIS
  idx_t ncon = 1;

  // Prepare remaining arguments for ParMETIS
  idx_t* elmwgt = NULL;
  idx_t wgtflag = 0;
  idx_t edgecut = 0;
  idx_t numflag = 0;
  std::vector<real_t> tpwgts(ncon*nparts, 1.0/static_cast<real_t>(nparts));
  std::vector<real_t> ubvec(ncon, 1.05);

  // Call ParMETIS to partition graph
  Timer timer1("ParMETIS: call ParMETIS_V3_PartKway");
  const std::int32_t num_local_cells = csr_graph.size();
  std::vector<idx_t> part(num_local_cells);
  dolfin_assert(!part.empty());
  int err
    = ParMETIS_V3_PartKway(csr_graph.node_distribution().data(),
                           csr_graph.nodes().data(),
                           csr_graph.edges().data(), elmwgt,
                           NULL, &wgtflag, &numflag, &ncon, &nparts,
                           tpwgts.data(), ubvec.data(), options,
                           &edgecut, part.data(),
                           &mpi_comm);
  dolfin_assert(err == METIS_OK);
  timer1.stop();

  Timer timer2("Compute graph halo data (ParMETIS)");

  // Work out halo cells for current division of dual graph
  const auto& elmdist = csr_graph.node_distribution();
  const auto& xadj = csr_graph.nodes();
  const auto& adjncy = csr_graph.edges();
  const std::int32_t num_processes = MPI::size(mpi_comm);
  const std::int32_t process_number = MPI::rank(mpi_comm);
  const idx_t elm_begin = elmdist[process_number];
  const idx_t elm_end = elmdist[process_number + 1];
  const std::int32_t ncells = elm_end - elm_begin;

  std::map<idx_t, std::set<std::int32_t>> halo_cell_to_remotes;
  // local indexing "i"
  for(int i = 0; i < ncells; i++)
  {
    for(auto other_cell : csr_graph[i]) //idx_t j = xadj[i]; j != xadj[i + 1]; ++j)
    {
      //      const idx_t other_cell = adjncy[j];
      if (other_cell < elm_begin || other_cell >= elm_end)
      {
        const int remote = std::upper_bound(elmdist.begin(),
                                            elmdist.end(),
                                            other_cell) - elmdist.begin() - 1;
        dolfin_assert(remote < num_processes);
        if (halo_cell_to_remotes.find(i) == halo_cell_to_remotes.end())
          halo_cell_to_remotes[i] = std::set<std::int32_t>();
        halo_cell_to_remotes[i].insert(remote);
      }
    }
  }

  // Do halo exchange of cell partition data
  std::vector<std::vector<std::int64_t>> send_cell_partition(num_processes);
  std::vector<std::vector<std::int64_t>> recv_cell_partition(num_processes);
  for(const auto& hcell : halo_cell_to_remotes)
  {
    for(auto proc : hcell.second)
    {
      dolfin_assert(proc < num_processes);
      // global cell number
      send_cell_partition[proc].push_back(hcell.first + elm_begin);
      //partitioning
      send_cell_partition[proc].push_back(part[hcell.first]);
    }
  }

  // Actual halo exchange
  MPI::all_to_all(mpi_comm, send_cell_partition, recv_cell_partition);

  // Construct a map from all currently foreign cells to their new
  // partition number
  std::map<std::int64_t, std::int32_t> cell_ownership;
  for (std::int32_t i = 0; i < num_processes; ++i)
  {
    std::vector<std::int64_t>& recv_data = recv_cell_partition[i];
    for (std::int32_t j = 0; j != recv_data.size(); j += 2)
    {
      const std::int64_t global_cell = recv_data[j];
      const std::int32_t cell_owner = recv_data[j+1];
      cell_ownership[global_cell] = cell_owner;
    }
  }

  // Generate mapping for where new boundary cells need to be sent
  for(std::int32_t i = 0; i < ncells; i++)
  {
    const std::size_t proc_this = part[i];
    for (idx_t j = xadj[i]; j != xadj[i + 1]; ++j)
    {
      const idx_t other_cell = adjncy[j];
      std::size_t proc_other;

      if (other_cell < elm_begin || other_cell >= elm_end)
      { // remote cell - should be in map
        const auto find_other_proc = cell_ownership.find(other_cell);
        dolfin_assert(find_other_proc != cell_ownership.end());
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
          ghost_procs.insert(std::make_pair(i, sharing_processes));
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

  // Copy cell partition data
  cell_partition.assign(part.begin(), part.end());
}
//-----------------------------------------------------------------------------
template <typename T>
void ParMETIS::adaptive_repartition(MPI_Comm mpi_comm,
                                    CSRGraph<T>& csr_graph,
                                    std::vector<int>& cell_partition)
{
  Timer timer("Compute graph partition (ParMETIS Adaptive Repartition)");

  // Options for ParMETIS
  idx_t options[4];
  options[0] = 1;
  options[1] = 0;
  options[2] = 15;
  options[3] = PARMETIS_PSR_UNCOUPLED;
  // For repartition, PARMETIS_PSR_COUPLED seems to suppress all
  // migration if already balanced.  Try PARMETIS_PSR_UNCOUPLED for
  // better edge cut.


  Timer timer1("ParMETIS: call ParMETIS_V3_AdaptiveRepart");
  const double itr = parameters["ParMETIS_repartitioning_weight"];
  real_t _itr = itr;
  std::vector<idx_t> part(csr_graph.size());
  std::vector<idx_t> vsize(part.size(), 1);
  dolfin_assert(!part.empty());

  // Number of partitions (one for each process)
  idx_t nparts = MPI::size(mpi_comm);
  // Remaining ParMETIS parameters
  idx_t ncon = 1;
  idx_t* elmwgt = NULL;
  idx_t wgtflag = 0;
  idx_t edgecut = 0;
  idx_t numflag = 0;
  std::vector<real_t> tpwgts(ncon*nparts, 1.0/static_cast<real_t>(nparts));
  std::vector<real_t> ubvec(ncon, 1.05);

  // Call ParMETIS to repartition graph
  int err = ParMETIS_V3_AdaptiveRepart(csr_graph.node_distribution().data(),
                                       csr_graph.nodes().data(),
                                       csr_graph.edges().data(),
                                       elmwgt, NULL, vsize.data(), &wgtflag,
                                       &numflag, &ncon, &nparts,
                                       tpwgts.data(), ubvec.data(), &_itr,
                                       options, &edgecut, part.data(),
                                       &mpi_comm);
  dolfin_assert(err == METIS_OK);
  timer1.stop();

  // Copy cell partition data
  cell_partition.assign(part.begin(), part.end());
}
//-----------------------------------------------------------------------------
template<typename T>
void ParMETIS::refine(MPI_Comm mpi_comm,
                      CSRGraph<T>& csr_graph,
                      std::vector<int>& cell_partition)
{
  Timer timer("Compute graph partition (ParMETIS Refine)");

  // Get some MPI data
  const std::size_t process_number = MPI::rank(mpi_comm);

  // Options for ParMETIS
  idx_t options[4];
  options[0] = 1;
  options[1] = 0;
  options[2] = 15;
  //options[3] = PARMETIS_PSR_UNCOUPLED;

  // For repartition, PARMETIS_PSR_COUPLED seems to suppress all
  // migration if already balanced.  Try PARMETIS_PSR_UNCOUPLED for
  // better edge cut.

  // Partitioning array to be computed by ParMETIS. Prefill with
  // process_number.
  const std::size_t num_local_cells = csr_graph.size();
  std::vector<idx_t> part(num_local_cells, process_number);
  dolfin_assert(!part.empty());

  // Number of partitions (one for each process)
  idx_t nparts = MPI::size(mpi_comm);
  // Remaining ParMETIS parameters
  idx_t ncon = 1;
  idx_t* elmwgt = NULL;
  idx_t wgtflag = 0;
  idx_t edgecut = 0;
  idx_t numflag = 0;
  std::vector<real_t> tpwgts(ncon*nparts, 1.0/static_cast<real_t>(nparts));
  std::vector<real_t> ubvec(ncon, 1.05);

  // Call ParMETIS to partition graph
  Timer timer1("ParMETIS: call ParMETIS_V3_RefineKway");
  int err =
    ParMETIS_V3_RefineKway(csr_graph.node_distribution().data(),
                           csr_graph.nodes().data(),
                           csr_graph.edges().data(),
                           elmwgt, NULL, &wgtflag, &numflag, &ncon,
                           &nparts,
                           tpwgts.data(), ubvec.data(), options,
                           &edgecut, part.data(), &mpi_comm);
  dolfin_assert(err == METIS_OK);
  timer1.stop();

  // Copy cell partition data
  cell_partition.assign(part.begin(), part.end());
}
//-----------------------------------------------------------------------------
CSRGraph<idx_t> ParMETIS::dual_graph(MPI_Comm mpi_comm,
                                     const boost::multi_array<std::int64_t, 2>& cell_vertices,
                                     const int num_vertices_per_cell)
{
  Timer timer("Build mesh dual graph (ParMETIS)");

  // ParMETIS data
  std::vector<idx_t> elmdist;
  std::vector<idx_t> eptr;
  std::vector<idx_t> eind;


  // Get number of processes and process number
  const std::size_t num_processes = MPI::size(mpi_comm);

  // Get dimensions of local mesh_data
  const std::size_t num_local_cells = cell_vertices.size();
  const std::size_t num_cell_vertices = num_vertices_per_cell;

  // Check that number of local graph nodes (cells) is > 0
  if (num_local_cells == 0)
  {
    dolfin_error("ParMETIS.cpp",
                 "compute mesh partitioning using ParMETIS",
                 "ParMETIS cannot be used if a process has no cells (graph nodes). Use SCOTCH to perform partitioning instead");
  }

  // Communicate number of cells between all processors
  std::vector<std::size_t> num_cells;
  MPI::all_gather(mpi_comm, num_local_cells, num_cells);

  // Build elmdist array with cell offsets for all processors
  elmdist.assign(num_processes + 1, 0);
  for (std::size_t i = 1; i < num_processes + 1; ++i)
    elmdist[i] = elmdist[i - 1] + num_cells[i - 1];

  eptr.resize(num_local_cells + 1);
  eind.assign(num_local_cells*num_cell_vertices, 0);
  for (std::size_t i = 0; i < num_local_cells; i++)
  {
    dolfin_assert(cell_vertices[i].size() == num_cell_vertices);
    eptr[i] = i*num_cell_vertices;
    for (std::size_t j = 0; j < num_cell_vertices; j++)
      eind[eptr[i] + j] = cell_vertices[i][j];
  }
  eptr[num_local_cells] = num_local_cells*num_cell_vertices;

  dolfin_assert(!eptr.empty());
  dolfin_assert(!eind.empty());

  // Number of nodes shared for dual graph (partition along facets)
  idx_t ncommonnodes = num_cell_vertices - 1;

  dolfin_assert(!eptr.empty());
  dolfin_assert(!eind.empty());

  // Could use GraphBuilder::compute_dual_graph() instead
  Timer timer1("ParMETIS: call ParMETIS_V3_Mesh2Dual");
  idx_t* xadj = NULL;
  idx_t* adjncy = NULL;
  idx_t numflag = 0;
  int err = ParMETIS_V3_Mesh2Dual(elmdist.data(), eptr.data(), eind.data(),
                                  &numflag, &ncommonnodes,
                                  &xadj, &adjncy,
                                  &mpi_comm);
  dolfin_assert(err == METIS_OK);
  timer1.stop();

  CSRGraph<idx_t> csr_graph(mpi_comm, xadj, adjncy, num_local_cells);
  METIS_Free(xadj);
  METIS_Free(adjncy);

  return csr_graph;
}
//-----------------------------------------------------------------------------
#else
void ParMETIS::compute_partition(const MPI_Comm mpi_comm,
                                 std::vector<int>& cell_partition,
                                 std::map<std::int64_t, std::vector<int>>& ghost_procs,
                                 const boost::multi_array<std::int64_t, 2>& cell_vertices,
                                 const int num_vertices_per_cell,
                                 const std::string mode)
{
  dolfin_error("ParMETIS.cpp",
               "compute mesh partitioning using ParMETIS",
               "DOLFIN has been configured without support for ParMETIS");
}
//-----------------------------------------------------------------------------
#endif
