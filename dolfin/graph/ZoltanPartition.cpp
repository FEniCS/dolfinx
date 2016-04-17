// Copyright (C) 2013 Chris Richardson
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
// First added:  2013-02-13
// Last changed: 2013-02-26

#include<set>
#include<string>
#include<vector>

#include <dolfin/common/MPI.h>
#include <dolfin/common/Timer.h>
#include <dolfin/log/log.h>
#include <dolfin/mesh/LocalMeshData.h>
#include <dolfin/parameter/GlobalParameters.h>
#include "GraphBuilder.h"
#include "ZoltanPartition.h"

using namespace dolfin;

#ifdef HAS_TRILINOS

//-----------------------------------------------------------------------------
void
ZoltanPartition::compute_partition_phg(const MPI_Comm mpi_comm,
                                       std::vector<int>& cell_partition,
                                       const LocalMeshData& mesh_data)
{
  Timer timer0("Partition graph (calling Zoltan PHG)");

  // Create data structures to hold graph
  std::vector<std::set<std::size_t>> local_graph;
  std::set<std::size_t> ghost_vertices;

  // Compute local dual graph
  GraphBuilder::compute_dual_graph(mpi_comm, mesh_data, local_graph,
                                   ghost_vertices);

  // Initialise Zoltan
  float version;
  int argc = 0;
  char** argv = NULL;
  Zoltan_Initialize(argc, argv, &version);

  // Create Zoltan object
  Zoltan zoltan;

  // Set Zoltan parameters
  zoltan.Set_Param("NUM_GID_ENTRIES", "1");
  zoltan.Set_Param("NUM_LID_ENTRIES", "0");

  zoltan.Set_Param("NUM_GLOBAL_PARTS",
                   std::to_string(MPI::size(mpi_comm)));

  zoltan.Set_Param("NUM_LOCAL_PARTS", "1");
  zoltan.Set_Param("LB_METHOD", "GRAPH");

  // Get partition method: 'PARTITION', 'REPARTITION' or 'REFINE'
  std::string lb_approach = parameters["partitioning_approach"];
  zoltan.Set_Param("LB_APPROACH", lb_approach.c_str());

  // Repartitioning weighting
  double phg_repart_multiplier = parameters["Zoltan_PHG_REPART_MULTIPLIER"];
  zoltan.Set_Param("PHG_REPART_MULTIPLIER",
                   std::to_string(phg_repart_multiplier));


  // Set call-back functions
  void *mesh_data_ptr = (void *)&mesh_data;
  zoltan.Set_Num_Obj_Fn(get_number_of_objects, mesh_data_ptr);
  zoltan.Set_Obj_List_Fn(get_object_list, mesh_data_ptr);

  void *graph_data_ptr = (void *)&local_graph;
  zoltan.Set_Num_Edges_Multi_Fn(get_number_edges, graph_data_ptr);
  zoltan.Set_Edge_List_Multi_Fn(get_all_edges, graph_data_ptr);

  // Call Zoltan function to compute partitions
  int changes = 0;
  int num_gids = 0;
  int num_lids = 0;
  int num_import, num_export;
  ZOLTAN_ID_PTR import_lids;
  ZOLTAN_ID_PTR export_lids;
  ZOLTAN_ID_PTR import_gids;
  ZOLTAN_ID_PTR export_gids;
  int* import_procs;
  int* export_procs;
  int* import_parts;
  int* export_parts;

  int rc = zoltan.LB_Partition(changes, num_gids, num_lids,
           num_import, import_gids, import_lids, import_procs, import_parts,
           num_export, export_gids, export_lids, export_procs, export_parts);

  dolfin_assert(num_gids == 1);
  dolfin_assert(num_lids == 0);

  std::size_t proc = MPI::rank(mpi_comm);

  if (rc != ZOLTAN_OK)
  {
    dolfin_error("ZoltanPartition.cpp",
                 "partition mesh using Zoltan",
                 "Call to Zoltan failed");
  }

  cell_partition.assign(local_graph.size(), proc);

  std::size_t offset = MPI::global_offset(mpi_comm, local_graph.size(), true);
  for(int i = 0; i < num_export; ++i)
  {
    const std::size_t idx = export_gids[i] - offset;
    cell_partition[idx] = (int)export_procs[i];
  }

  // Free data structures allocated by Zoltan::LB_Partition
  zoltan.LB_Free_Part(&import_gids, &import_lids, &import_procs, &import_parts);
  zoltan.LB_Free_Part(&export_gids, &export_lids, &export_procs, &export_parts);
}
//-----------------------------------------------------------------------------
void
ZoltanPartition::compute_partition_rcb(const MPI_Comm mpi_comm,
                                       std::vector<int>& cell_partition,
                                       const LocalMeshData& mesh_data)
{
  Timer timer0("Partition graph (calling Zoltan RCB)");

  // Get number of local graph vertices
  const std::size_t nlocal = mesh_data.cell_vertices.shape()[0];

  // Initialise Zoltan
  float version;
  int argc = 0;
  char** argv = NULL;
  Zoltan_Initialize(argc, argv, &version);

  // Create Zoltan object
  Zoltan zoltan;

  // Set Zoltan parameters
  zoltan.Set_Param("NUM_GID_ENTRIES", "1");
  zoltan.Set_Param("NUM_LID_ENTRIES", "0");

  zoltan.Set_Param("NUM_GLOBAL_PARTS",
                   std::to_string(MPI::size(mpi_comm)));

  zoltan.Set_Param("NUM_LOCAL_PARTS", "1");
  zoltan.Set_Param("LB_METHOD", "RCB");

  // Set call-back functions
  void *mesh_data_ptr = (void *)&mesh_data;

  zoltan.Set_Num_Obj_Fn(get_number_of_objects, mesh_data_ptr);
  zoltan.Set_Obj_List_Fn(get_object_list, mesh_data_ptr);
  zoltan.Set_Num_Geom_Fn(get_geom, mesh_data_ptr);
  zoltan.Set_Geom_Multi_Fn(get_all_geom, mesh_data_ptr);

  // Call Zoltan function to compute partitions
  int changes = 0;
  int num_gids = 0;
  int num_lids = 0;
  int num_import, num_export;
  ZOLTAN_ID_PTR import_lids;
  ZOLTAN_ID_PTR export_lids;
  ZOLTAN_ID_PTR import_gids;
  ZOLTAN_ID_PTR export_gids;
  int* import_procs;
  int* export_procs;
  int* import_parts;
  int* export_parts;

  int rc = zoltan.LB_Partition(changes, num_gids, num_lids,
           num_import, import_gids, import_lids, import_procs, import_parts,
           num_export, export_gids, export_lids, export_procs, export_parts);

  dolfin_assert(num_gids == 1);
  dolfin_assert(num_lids == 0);


  // Get my process rank
  const std::size_t my_rank = MPI::rank(mpi_comm);

  if (rc != ZOLTAN_OK)
  {
    dolfin_error("ZoltanPartition.cpp",
                 "partition mesh using Zoltan",
                 "Call to Zoltan failed");
  }

  // Assign all nodes to this processor
  cell_partition.assign(nlocal, my_rank);
  std::size_t offset = MPI::global_offset(mpi_comm, nlocal, true);

  // Change nodes to be exported to the appropriate remote processor
  for(int i = 0; i < num_export; ++i)
  {
    const std::size_t idx = export_gids[i] - offset;
    cell_partition[idx] = export_procs[i];
  }

  // Free data structures allocated by Zoltan::LB_Partition
  zoltan.LB_Free_Part(&import_gids, &import_lids, &import_procs, &import_parts);
  zoltan.LB_Free_Part(&export_gids, &export_lids, &export_procs, &export_parts);
}
//-----------------------------------------------------------------------------
int ZoltanPartition::get_number_of_objects(void* data, int* ierr)
{
  LocalMeshData* local_mesh_data = (LocalMeshData*)data;
  *ierr = ZOLTAN_OK;
  return local_mesh_data->cell_vertices.shape()[0];
}
//-----------------------------------------------------------------------------
void ZoltanPartition::get_object_list(void *data,
                                      int num_gid_entries,
                                      int num_lid_entries,
                                      ZOLTAN_ID_PTR global_id,
                                      ZOLTAN_ID_PTR local_id, int wgt_dim,
                                      float* obj_wgts, int* ierr)
{
  LocalMeshData* local_mesh_data = (LocalMeshData*)data;
  dolfin_assert(local_mesh_data);

  // Get MPI communicator
  const MPI_Comm mpi_comm = local_mesh_data->mpi_comm();

  dolfin_assert(num_gid_entries == 1);
  dolfin_assert(num_lid_entries == 0);

  const std::size_t nlocal = local_mesh_data->cell_vertices.shape()[0];
  const std::size_t offset = MPI::global_offset(mpi_comm, nlocal, true);

  for (std::size_t i = 0; i < nlocal; ++i)
    global_id[i] = i + offset;

  dolfin_assert(wgt_dim == 0);
  obj_wgts = NULL;

  *ierr = ZOLTAN_OK;
}
//-----------------------------------------------------------------------------
void ZoltanPartition::get_number_edges(void *data,
                                       int num_gid_entries,
                                       int num_lid_entries,
                                       int num_obj, ZOLTAN_ID_PTR global_ids,
                                       ZOLTAN_ID_PTR local_ids, int *num_edges,
                                       int *ierr)
{
  std::vector<std::set<std::size_t>>* local_graph
    = (std::vector<std::set<std::size_t>>*)data;

  dolfin_assert(num_gid_entries == 1);
  dolfin_assert(num_lid_entries == 0);
  dolfin_assert(num_obj == (int)local_graph->size());

  for (std::size_t i = 0; i < local_graph->size(); ++i)
    num_edges[i] = (*local_graph)[i].size();

  *ierr = ZOLTAN_OK;
}
//-----------------------------------------------------------------------------
void ZoltanPartition::get_all_edges(void* data,
                                 int num_gid_entries,
                                 int num_lid_entries, int num_obj,
                                 ZOLTAN_ID_PTR global_ids,
                                 ZOLTAN_ID_PTR local_ids,
                                 int* num_edges,
                                 ZOLTAN_ID_PTR nbor_global_id,
                                 int* nbor_procs, int wgt_dim,
                                 float* ewgts, int* ierr)
{
  // FIXME: ZoltanPartition::get_all_edges need to be updated for MPI communicator
  dolfin_not_implemented();

  /*
  // Get graph
  const std::vector<std::set<std::size_t>>* local_graph
    = (std::vector<std::set<std::size_t>>*)data;

  // MPI communicate
  const MPI_Comm mpi_comm = local_graph->mpi_comm();

  std::vector<std::size_t> offsets;
  std::size_t local_offset = MPI::global_offset(mpi_comm, local_graph->size(),
                                                true);
  MPI::all_gather(mpi_comm, local_offset, offsets);
  offsets.push_back(MPI::sum(mpi_comm, local_graph->size()));

  std::size_t i = 0;
  for(std::vector<std::set<std::size_t>>::iterator node = local_graph->begin();
      node != local_graph->end(); ++node)
  {
    for(std::set<std::size_t>::iterator edge = node->begin();
        edge != node->end(); ++edge)
    {
      nbor_global_id[i] = *edge;
      nbor_procs[i] = std::upper_bound(offsets.begin(), offsets.end(), *edge)
        - offsets.begin()-1;
      i++;
    }
  }

  dolfin_assert(wgt_dim == 0);
  ewgts = NULL;
  *ierr = ZOLTAN_OK;
  */
}
//-----------------------------------------------------------------------------
int ZoltanPartition::get_geom(void* data, int* ierr)
{
  LocalMeshData* local_mesh_data=(LocalMeshData*)data;

  *ierr = ZOLTAN_OK;
  return local_mesh_data->gdim;
}
//-----------------------------------------------------------------------------
void ZoltanPartition::get_all_geom(void *data,
                                   int num_gid_entries, int num_lid_entries,
                                   int num_obj, ZOLTAN_ID_PTR global_ids,
                                   ZOLTAN_ID_PTR local_ids,
                                   int num_dim, double *geom_vec, int *ierr)
{
  const LocalMeshData* local_mesh_data=(LocalMeshData*)data;
  dolfin_assert(local_mesh_data);

  // Get MPI communicator
  const MPI_Comm mpi_comm = local_mesh_data->mpi_comm();

  dolfin_assert(num_gid_entries == 1);
  dolfin_assert(num_lid_entries == 0);

  std::size_t gdim = local_mesh_data->gdim;
  dolfin_assert(num_dim == (int)gdim);

  std::size_t num_local_cells = local_mesh_data->cell_vertices.shape()[0];
  std::size_t cell_offset = MPI::global_offset(mpi_comm, num_local_cells, true);
  std::size_t num_vertices_per_cell = local_mesh_data->cell_vertices.shape()[1];
  dolfin_assert(num_obj == (int)num_local_cells);

  // Work out all ranges for vertices
  std::size_t num_local_vertices
    = local_mesh_data->vertex_coordinates.shape()[0];
  std::vector<std::size_t> vertex_offsets;
  std::size_t local_vertex_offset
    = MPI::global_offset(mpi_comm, num_local_vertices, true);
  MPI::all_gather(mpi_comm, local_vertex_offset, vertex_offsets);
  vertex_offsets.push_back(MPI::sum(mpi_comm, num_local_vertices));

  // Need to get all vertex coordinates which are referred to by
  // topology onto this process...
  std::map<std::size_t, std::vector<double>> vertex;

  // Insert local vertices into map
  for(std::size_t i = 0; i < num_local_vertices; ++i)
  {
    vertex[i + local_vertex_offset]
      = std::vector<double>(local_mesh_data->vertex_coordinates[i].begin(),
                            local_mesh_data->vertex_coordinates[i].end());
  }

  std::size_t num_processes = MPI::size(mpi_comm);
  std::vector<std::vector<std::size_t>>send_buffer(num_processes);
  std::vector<std::vector<std::size_t>>receive_buffer(num_processes);

  // Make list of requested remote vertices for each remote process
  for (std::size_t i = 0; i < num_local_cells; ++i)
    for (std::size_t j = 0; j < num_vertices_per_cell; ++j)
    {
      std::size_t ivtx = local_mesh_data->cell_vertices[i][j];
      if (ivtx < local_vertex_offset
         || ivtx >= (local_vertex_offset + num_local_vertices))
      {
        // Find owner of this vertex and then add to request list for
        // that process
        std::size_t proc = std::upper_bound(vertex_offsets.begin(),
                                            vertex_offsets.end(), ivtx)
          - vertex_offsets.begin() - 1;
        send_buffer[proc].push_back(ivtx);
      }
    }

  MPI::all_to_all(mpi_comm, send_buffer, receive_buffer);

  std::vector<std::vector<double>> dsend_buffer(num_processes);
  std::vector<std::vector<double>> dreceive_buffer(num_processes);

  // Get received requests for vertices from remote processes, and put
  // together answer to send back
  for (std::vector<std::vector<std::size_t>>::iterator p
         = receive_buffer.begin(); p != receive_buffer.end(); ++p)
  {
    std::size_t proc = p - receive_buffer.begin();
    for (std::vector<std::size_t>::iterator vidx = p->begin();
         vidx != p->end(); ++vidx)
    {
      dolfin_assert(*vidx >= local_vertex_offset);
      dolfin_assert(*vidx < (local_vertex_offset + num_local_vertices));
      dsend_buffer[proc].insert(dsend_buffer[proc].end(),
                                local_mesh_data->vertex_coordinates[*vidx - local_vertex_offset].begin(),
                                local_mesh_data->vertex_coordinates[*vidx - local_vertex_offset].end());
    }
  }

  MPI::all_to_all(mpi_comm, dsend_buffer, dreceive_buffer);

  // insert received coordinates into local map.
  for(std::size_t i = 0; i < num_processes; ++i)
  {
    std::vector<double>::iterator vcoords = dreceive_buffer[i].begin();
    for(std::vector<std::size_t>::iterator v = send_buffer[i].begin();
        v != send_buffer[i].end(); ++v)
    {
      vertex[*v] = std::vector<double>(vcoords, vcoords + gdim);
      vcoords += gdim;
    }
  }

  double *geom_ptr = geom_vec;
  for (std::size_t i = 0; i < num_local_cells; ++i)
  {
    std::vector<double> x(gdim, 0.0);
    for(std::size_t j = 0; j < num_vertices_per_cell ; ++j)
    {
      std::size_t idx
        = local_mesh_data->cell_vertices[global_ids[i] - cell_offset][j] ;
      for(std::size_t k=0; k < gdim ; ++k)
        x[k] += vertex[idx][k];
    }

    std::copy(x.begin(), x.end(), geom_ptr);
    geom_ptr += gdim;
  }

  *ierr = ZOLTAN_OK;
}
//-----------------------------------------------------------------------------
#else
void ZoltanPartition::compute_partition_phg(const MPI_Comm mpi_comm,
                                            std::vector<int>& cell_partition,
                                            const LocalMeshData& mesh_data)
{
  dolfin_error("ZoltanPartition.cpp",
               "partition mesh using Zoltan",
               "DOLFIN has been configured without support for Zoltan from Trilinos");
}
//-----------------------------------------------------------------------------
void ZoltanPartition::compute_partition_rcb(const MPI_Comm mpi_comm,
                                            std::vector<int>& cell_partition,
                                            const LocalMeshData& mesh_data)
{
  dolfin_error("ZoltanPartition.cpp",
               "partition mesh using Zoltan",
               "DOLFIN has been configured without support for Zoltan from Trilinos");
}
#endif
//-----------------------------------------------------------------------------
