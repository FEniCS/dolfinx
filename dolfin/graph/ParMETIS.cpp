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
// Last changed: 2013-02-26

#include <dolfin/log/dolfin_log.h>

#include <dolfin/common/Timer.h>
#include <dolfin/common/MPI.h>
#include <dolfin/parameter/GlobalParameters.h>
#include <dolfin/mesh/LocalMeshData.h>
#include "ParMETIS.h"
#include "GraphBuilder.h"

#ifdef HAS_PARMETIS
#include <parmetis.h>
#endif

using namespace dolfin;

#ifdef HAS_PARMETIS
//-----------------------------------------------------------------------------
void ParMETIS::compute_partition(std::vector<std::size_t>& cell_partition,
                                 const LocalMeshData& mesh_data)
{
  const std::string approach = parameters["partitioning_approach"];
  
  if(approach == "PARTITION")
    partition(cell_partition, mesh_data);
  else
    repartition(cell_partition, mesh_data);
}


void ParMETIS::partition(std::vector<std::size_t>& cell_partition,
                         const LocalMeshData& mesh_data)
{
  // This function prepares data for ParMETIS calls ParMETIS, and then
  // collects the results from ParMETIS.

  Timer timer0("PARALLEL 1a: Build distributed dual graph (calling ParMETIS)");

  // Get number of processes and process number
  const std::size_t num_processes = MPI::num_processes();

  // Get dimensions of local mesh_data
  const std::size_t num_local_cells = mesh_data.cell_vertices.size();
  const std::size_t num_cell_vertices = mesh_data.num_vertices_per_cell;

  // Check that number of local graph nodes (cells) is > 0
  if (num_local_cells == 0)
  {
    dolfin_error("ParMETIS.cpp",
                 "compute mesh partitioning using ParMETIS",
                 "ParMETIS cannot be used if a process has no cells (graph nodes). Use SCOTCH to perform partitioning instead");
  }

  // Communicate number of cells between all processors
  std::vector<std::size_t> num_cells;
  MPI::all_gather(num_local_cells, num_cells);

  // Build elmdist array with cell offsets for all processors
  std::vector<int> elmdist(num_processes + 1, 0);
  for (std::size_t i = 1; i < num_processes + 1; ++i)
    elmdist[i] = elmdist[i - 1] + num_cells[i - 1];

  // Build eptr and eind arrays storing cell-vertex connectivity
  std::vector<int> eptr(num_local_cells + 1);
  std::vector<int> eind(num_local_cells*num_cell_vertices, 0);
  for (std::size_t i = 0; i < num_local_cells; i++)
  {
    dolfin_assert(mesh_data.cell_vertices[i].size() == num_cell_vertices);
    eptr[i] = i*num_cell_vertices;
    for (std::size_t j = 0; j < num_cell_vertices; j++)
      eind[eptr[i] + j] = mesh_data.cell_vertices[i][j];
  }
  eptr[num_local_cells] = num_local_cells*num_cell_vertices;

  // Number of nodes shared for dual graph (partition along facets)
  int ncommonnodes = num_cell_vertices - 1;

  // Number of partitions (one for each process)
  int nparts = num_processes;

  // Strange weight arrays needed by ParMETIS
  int ncon = 1;
  #if PARMETIS_MAJOR_VERSION >= 4
  std::vector<real_t> tpwgts(ncon*nparts, 1.0/static_cast<real_t>(nparts));
  std::vector<real_t> ubvec(ncon, 1.05);
  #else
  std::vector<float> tpwgts(ncon*nparts, 1.0/static_cast<float>(nparts));
  std::vector<float> ubvec(ncon, 1.05);
  #endif

  // Options for ParMETIS, use default
  int options[3];
  options[0] = 1;
  options[1] = 0;
  options[2] = 15;

  // Partitioning array to be computed by ParMETIS (note bug in manual:
  // vertices, not cells!)
  std::vector<int> part(num_local_cells);

  // Prepare remaining arguments for ParMETIS
  int* elmwgt = 0;
  int wgtflag = 0;
  int numflag = 0;
  int edgecut = 0;

  dolfin_assert(!elmdist.empty());
  dolfin_assert(!eptr.empty());
  dolfin_assert(!eind.empty());

  // Construct communicator (copy of MPI_COMM_WORLD)
  MPICommunicator comm;

  // Build dual graph from mesh
  idx_t* xadj = 0;
  idx_t* adjncy = 0;
  int err = ParMETIS_V3_Mesh2Dual(elmdist.data(), eptr.data(), eind.data(),
                                  &numflag, &ncommonnodes,
                                  &xadj, &adjncy,
                                  &(*comm));
  dolfin_assert(err == METIS_OK);

  timer0.stop();

  Timer timer1("PARALLEL 1b: Compute graph partition (calling ParMETIS)");

  // Call ParMETIS to partition graph
  dolfin_assert(!tpwgts.empty());
  dolfin_assert(!ubvec.empty());
  dolfin_assert(!part.empty());
  err = ParMETIS_V3_PartKway(elmdist.data(), xadj, adjncy, elmwgt,
                             NULL, &wgtflag, &numflag, &ncon, &nparts,
                             tpwgts.data(), ubvec.data(), options,
                             &edgecut, part.data(), &(*comm));
  dolfin_assert(err == METIS_OK);

  // Length of xadj = #local nodes + 1
  // Length of adjncy = xadj[-1]

  METIS_Free(xadj);
  METIS_Free(adjncy);

  // Copy cell partition data
  cell_partition = std::vector<std::size_t>(part.begin(), part.end());
}
//-----------------------------------------------------------------------------
void ParMETIS::repartition(std::vector<std::size_t>& cell_partition,
                           const LocalMeshData& mesh_data)
{
  // Recompute partitioning based on existing
  // Useful when refining mesh etc.

  // Get number of processes and process number
  const std::size_t num_processes = MPI::num_processes();

  // Get dimensions of local mesh_data
  const std::size_t num_local_cells = mesh_data.cell_vertices.size();

  // Check that number of local graph nodes (cells) is > 0
  if (num_local_cells == 0)
  {
    dolfin_error("ParMETIS.cpp",
                 "recompute mesh partitioning using ParMETIS",
                 "ParMETIS cannot be used if a process has no cells (graph nodes). Use SCOTCH to perform partitioning instead");
  }

  double tt = time();
  // Communicate number of cells between all processors
  std::vector<std::size_t> num_cells;
  MPI::all_gather(num_local_cells, num_cells);
  // Build elmdist array with cell offsets for all processors
  std::vector<int> elmdist(num_processes + 1, 0);
  for (std::size_t i = 1; i < num_processes + 1; ++i)
    elmdist[i] = elmdist[i - 1] + num_cells[i - 1];

  // Number of partitions (one for each process)
  int nparts = num_processes;

  // Strange weight arrays needed by ParMETIS
  int ncon = 1;
  #if PARMETIS_MAJOR_VERSION >= 4
  std::vector<real_t> tpwgts(ncon*nparts, 1.0/static_cast<real_t>(nparts));
  std::vector<real_t> ubvec(ncon, 1.05);
  #else
  std::vector<float> tpwgts(ncon*nparts, 1.0/static_cast<float>(nparts));
  std::vector<float> ubvec(ncon, 1.05);
  #endif

  // Options for ParMETIS, use default
  int options[4];
  options[0] = 1;
  options[1] = 0;
  options[2] = 15;
  options[3] = PARMETIS_PSR_COUPLED;

  std::vector<int> part(num_local_cells);

  // Prepare remaining arguments for ParMETIS
  int* elmwgt = 0;
  int wgtflag = 0;
  int numflag = 0;
  int edgecut = 0;

  // Construct communicator (copy of MPI_COMM_WORLD)
  MPICommunicator comm;

  std::vector<std::set<std::size_t> > local_graph;
  std::set<std::size_t> ghost_vertices;

  // Calculate the dual graph for existing mesh
  GraphBuilder::compute_dual_graph(mesh_data, local_graph, ghost_vertices);
  std::vector<idx_t> adjncy;
  std::vector<idx_t> xadj;
  xadj.reserve(num_local_cells + 1);
  xadj.push_back(0);

  for(std::vector<std::set<std::size_t> >::iterator cell_c = local_graph.begin();
      cell_c != local_graph.end(); ++cell_c)
  {
    adjncy.insert(adjncy.end(), cell_c->begin(), cell_c->end());
    xadj.push_back(adjncy.size());
  }

  tt = time() - tt;
  double tt_max = MPI::max(tt);
  if (MPI::process_number() == 0)
    info("Time to build (parallel) dual graph: %g", tt_max);

  Timer timer1("PARALLEL 1b: Recompute graph partition (calling ParMETIS)");

  const double itr = parameters["ParMETIS_repartitioning_weight"];
  real_t _itr = itr;
  std::vector<idx_t> vsize(xadj.size(), 1);

  // Call ParMETIS to repartition graph
  dolfin_assert(!tpwgts.empty());
  dolfin_assert(!ubvec.empty());
  dolfin_assert(!part.empty());
  int err = ParMETIS_V3_AdaptiveRepart(elmdist.data(), xadj.data(), adjncy.data(), elmwgt,
                                       NULL, vsize.data(), &wgtflag, &numflag, &ncon, &nparts,
                                       tpwgts.data(), ubvec.data(), &_itr, options,
                                       &edgecut, part.data(), &(*comm));
  dolfin_assert(err == METIS_OK);

  // Length of xadj = # local nodes + 1
  // Length of adjncy = xadj[-1]

  //info("Partitioned mesh, edge cut is %d.", edgecut);

  // Copy cell partition data
  cell_partition = std::vector<std::size_t>(part.begin(), part.end());
}
//-----------------------------------------------------------------------------
#else
void ParMETIS::compute_partition(std::vector<std::size_t>& cell_partition,
                                    const LocalMeshData& data)
{
  dolfin_error("ParMETIS.cpp",
               "compute mesh partitioning using ParMETIS",
               "DOLFIN has been configured without support for ParMETIS");
}
//-----------------------------------------------------------------------------
#endif
