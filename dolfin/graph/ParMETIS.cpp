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
// Last changed: 2013-04-26

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

namespace dolfin
{
  // This class builds a ParMETIS dual graph

  class ParMETISDualGraph
  {
  public:

    // Constructor
    ParMETISDualGraph(const LocalMeshData& mesh_data);

    // Destructor
    ~ParMETISDualGraph();

    // ParMETIS data
    std::vector<int> elmdist;
    std::vector<int> eptr;
    std::vector<int> eind;
    int numflag;
    idx_t* xadj;
    idx_t* adjncy;

    // Number of partitions (one for each process)
    int nparts;

    // Strange weight arrays needed by ParMETIS
    int ncon;
    std::vector<real_t> tpwgts;
    std::vector<real_t> ubvec;

    // Prepare remaining arguments for ParMETIS
    int* elmwgt;
    int wgtflag;
    int edgecut;

  };
}
//-----------------------------------------------------------------------------
void ParMETIS::compute_partition(std::vector<std::size_t>& cell_partition,
                                 const LocalMeshData& mesh_data,
                                 std::string mode)
{
  // Build dual graph
  ParMETISDualGraph g(mesh_data);

  dolfin_assert(g.eptr.size() - 1 == mesh_data.cell_vertices.size());

  // Partition graph
  if (mode == "partition")
    partition(cell_partition, g);
  else if (mode == "adaptive_repartition")
    adaptive_repartition(cell_partition, g);
  else if (mode == "refine")
   refine(cell_partition, g);
  else
  {
    dolfin_error("ParMETIS.cpp",
                 "compute mesh partitioning using ParMETIS",
                 "partition model %s is unknown. Must be \"partition\", \"adactive_partition\" or \"refine\"", mode.c_str());
  }
}
//-----------------------------------------------------------------------------
void ParMETIS::partition(std::vector<std::size_t>& cell_partition,
                         ParMETISDualGraph& g)
{
  Timer timer1("PARALLEL 1b: Compute graph partition (calling ParMETIS)");

  // Options for ParMETIS
  int options[3];
  options[0] = 1;
  options[1] = 0;
  options[2] = 15;

  // Check that data arrays are not empty
  dolfin_assert(!g.tpwgts.empty());
  dolfin_assert(!g.ubvec.empty());

  // Construct communicator (copy of MPI_COMM_WORLD)
  MPICommunicator comm;

  // Call ParMETIS to partition graph
  const std::size_t num_local_cells = g.eptr.size() - 1;
  std::vector<int> part(num_local_cells);
  dolfin_assert(!part.empty());
  int err = ParMETIS_V3_PartKway(g.elmdist.data(), g.xadj, g.adjncy, g.elmwgt,
                                 NULL, &g.wgtflag, &g.numflag, &g.ncon, &g.nparts,
                                 g.tpwgts.data(), g.ubvec.data(), options,
                                 &g.edgecut, part.data(), &(*comm));
  dolfin_assert(err == METIS_OK);

  // Copy cell partition data
  cell_partition = std::vector<std::size_t>(part.begin(), part.end());
}
//-----------------------------------------------------------------------------
void ParMETIS::adaptive_repartition(std::vector<std::size_t>& cell_partition,
                                    ParMETISDualGraph& g)
{
  Timer timer1("PARALLEL 1b: Compute graph partition (calling ParMETIS Adaptive Repartition)");

  // Options for ParMETIS
  int options[4];
  options[0] = 1;
  options[1] = 0;
  options[2] = 15;
  options[3] = PARMETIS_PSR_UNCOUPLED;
  // For repartition, PARMETIS_PSR_COUPLED seems to suppress all migration if already balanced.
  // Try PARMETIS_PSR_UNCOUPLED for better edge cut.

  // Check that data arrays are not empty
  dolfin_assert(!g.tpwgts.empty());
  dolfin_assert(!g.ubvec.empty());

  // Construct communicator (copy of MPI_COMM_WORLD)
  MPICommunicator comm;

  // Call ParMETIS to partition graph
  const double itr = parameters["ParMETIS_repartitioning_weight"];
  real_t _itr = itr;
  std::vector<int> part(g.eptr.size() - 1);
  std::vector<idx_t> vsize(part.size(), 1);
  dolfin_assert(!part.empty());
  int err = ParMETIS_V3_AdaptiveRepart(g.elmdist.data(), g.xadj, g.adjncy,
                                       g.elmwgt, NULL, vsize.data(), &g.wgtflag,
                                       &g.numflag, &g.ncon, &g.nparts,
                                       g.tpwgts.data(), g.ubvec.data(), &_itr,
                                       options, &g.edgecut, part.data(),
                                       &(*comm));
  dolfin_assert(err == METIS_OK);

  // Copy cell partition data
  cell_partition = std::vector<std::size_t>(part.begin(), part.end());
}
//-----------------------------------------------------------------------------
void ParMETIS::refine(std::vector<std::size_t>& cell_partition,
                      ParMETISDualGraph& g)
{
  Timer timer1("PARALLEL 1b: Compute graph partition (calling ParMETIS Refine)");

  // Get some MPI data
  const std::size_t process_number = MPI::process_number();

  // Options for ParMETIS
  int options[4];
  options[0] = 1;
  options[1] = 0;
  options[2] = 15;
  //options[3] = PARMETIS_PSR_UNCOUPLED;
  // For repartition, PARMETIS_PSR_COUPLED seems to suppress all migration if already balanced.
  // Try PARMETIS_PSR_UNCOUPLED for better edge cut.

  // Check that data arrays are not empty
  dolfin_assert(!g.tpwgts.empty());
  dolfin_assert(!g.ubvec.empty());

  // Construct communicator (copy of MPI_COMM_WORLD)
  MPICommunicator comm;

  // Partitioning array to be computed by ParMETIS. Prefill with
  // process_number.
  const std::size_t num_local_cells = g.eptr.size() - 1;
  std::vector<int> part(num_local_cells, process_number);
  dolfin_assert(!part.empty());

  // Call ParMETIS to partition graph
  int err = ParMETIS_V3_RefineKway(g.elmdist.data(), g.xadj, g.adjncy, g.elmwgt,
                                   NULL, &g.wgtflag, &g.numflag, &g.ncon, &g.nparts,
                                   g.tpwgts.data(), g.ubvec.data(), options,
                                   &g.edgecut, part.data(), &(*comm));
  dolfin_assert(err == METIS_OK);

  // Copy cell partition data
  cell_partition = std::vector<std::size_t>(part.begin(), part.end());
}
//-----------------------------------------------------------------------------
ParMETISDualGraph::ParMETISDualGraph(const LocalMeshData& mesh_data)
{
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
  elmdist.assign(num_processes + 1, 0);
  for (std::size_t i = 1; i < num_processes + 1; ++i)
    elmdist[i] = elmdist[i - 1] + num_cells[i - 1];

  eptr.resize(num_local_cells + 1);
  eind.assign(num_local_cells*num_cell_vertices, 0);
  for (std::size_t i = 0; i < num_local_cells; i++)
  {
    dolfin_assert(mesh_data.cell_vertices[i].size() == num_cell_vertices);
    eptr[i] = i*num_cell_vertices;
    for (std::size_t j = 0; j < num_cell_vertices; j++)
      eind[eptr[i] + j] = mesh_data.cell_vertices[i][j];
  }
  eptr[num_local_cells] = num_local_cells*num_cell_vertices;

  dolfin_assert(!eptr.empty());
  dolfin_assert(!eind.empty());

  // Number of nodes shared for dual graph (partition along facets)
  int ncommonnodes = num_cell_vertices - 1;
  numflag = 0;
  xadj = 0;
  adjncy = 0;

  dolfin_assert(!eptr.empty());
  dolfin_assert(!eind.empty());

  // Construct communicator (copy of MPI_COMM_WORLD)
  MPICommunicator comm;

  // Could use GraphBuilder::compute_dual_graph() instead
  int err = ParMETIS_V3_Mesh2Dual(elmdist.data(), eptr.data(), eind.data(),
                                  &numflag, &ncommonnodes,
                                  &xadj, &adjncy,
                                  &(*comm));
  dolfin_assert(err == METIS_OK);


  // Number of partitions (one for each process)
  nparts = num_processes;

  // Strange weight arrays needed by ParMETIS
  ncon = 1;
  tpwgts.assign(ncon*nparts, 1.0/static_cast<real_t>(nparts));
  ubvec.assign(ncon, 1.05);

  // Prepare remaining arguments for ParMETIS
  elmwgt = NULL;
  wgtflag = 0;
  edgecut = 0;
}
//-----------------------------------------------------------------------------
ParMETISDualGraph::~ParMETISDualGraph()
{
  // Free metis data structures
  METIS_Free(xadj);
  METIS_Free(adjncy);
}
//-----------------------------------------------------------------------------
#else
void ParMETIS::compute_partition(std::vector<std::size_t>& cell_partition,
                                 const LocalMeshData& data,
                                 bool repartition)
{
  dolfin_error("ParMETIS.cpp",
               "compute mesh partitioning using ParMETIS",
               "DOLFIN has been configured without support for ParMETIS");
}
//-----------------------------------------------------------------------------
#endif
