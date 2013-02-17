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
// Last changed: 2013-02-17

#include<boost/lexical_cast.hpp>

#include <dolfin/common/MPI.h>
#include <dolfin/common/Timer.h>
#include <dolfin/mesh/LocalMeshData.h>

#include "GraphBuilder.h"
#include "ZoltanPartition.h"

using namespace dolfin;

#ifdef HAS_TRILINOS
//-----------------------------------------------------------------------------
void ZoltanPartition::num_vertex_edges(void * data, unsigned int* num_edges)
{
  dolfin_assert(num_edges);
}
//-----------------------------------------------------------------------------
int ZoltanPartition::get_number_of_objects(void* data, int* ierr)
{
  std::vector<std::set<std::size_t> > *local_graph 
    = (std::vector<std::set<std::size_t> >*)data;
  *ierr = ZOLTAN_OK;
  return local_graph->size();
}
//-----------------------------------------------------------------------------
void ZoltanPartition::get_object_list(void *data,
                                      int num_gid_entries,
                                      int num_lid_entries,
                                      ZOLTAN_ID_PTR global_id,
                                      ZOLTAN_ID_PTR local_id, int wgt_dim,
                                      float* obj_wgts, int* ierr)
{
  std::vector<std::set<std::size_t> > *local_graph 
    = (std::vector<std::set<std::size_t> >*)data;

  dolfin_assert(num_gid_entries == 1);
  dolfin_assert(num_lid_entries == 0);
  
  int offset = MPI::global_offset(local_graph->size(), true);
  
  for(std::size_t i = 0; i < local_graph->size(); ++i)
  {
    global_id[i] = i + offset;
  }

  std::cout << "WGT_DIM=" << wgt_dim << std::endl;
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
  std::vector<std::set<std::size_t> > *local_graph 
    = (std::vector<std::set<std::size_t> >*)data;

  dolfin_assert(num_gid_entries == 1);
  dolfin_assert(num_lid_entries == 0);
  dolfin_assert(num_obj == (int)local_graph->size());

  for(std::size_t i = 0; i < local_graph->size(); ++i)
  {
    num_edges[i] = (*local_graph)[i].size();
  }
  
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
  std::vector<std::set<std::size_t> > *local_graph 
    = (std::vector<std::set<std::size_t> >*)data;

  std::vector<std::size_t> offsets;
  std::size_t local_offset = MPI::global_offset(local_graph->size(), true);
  MPI::all_gather(local_offset, offsets);
  offsets.push_back(MPI::sum(local_graph->size()));

  std::size_t i = 0;
  for(std::vector<std::set<std::size_t> >::iterator node = local_graph->begin();
      node != local_graph->end(); ++node)
  {
    for(std::set<std::size_t>::iterator edge = node->begin(); edge != node->end(); ++edge)
    {
      nbor_global_id[i] = *edge;
      nbor_procs[i] = std::upper_bound(offsets.begin(), offsets.end(), *edge) - offsets.begin() - 1; 
      i++;
    }
  }

  dolfin_assert(wgt_dim == 0);
  ewgts = NULL;
  *ierr = ZOLTAN_OK;
  
}

void ZoltanPartition::compute_partition(std::vector<std::size_t>& cell_partition,
                                        const LocalMeshData& mesh_data)
{

  Timer timer0("Partition graph (calling Zoltan PHG)");

  // Create data structures to hold graph
  std::vector<std::set<std::size_t> > local_graph;
  std::set<std::size_t> ghost_vertices;
  // Compute local dual graph
  GraphBuilder::compute_dual_graph(mesh_data, local_graph, ghost_vertices);

  // Initialise Zoltan
  float version;
  int argc = 0;
  char** argv = NULL;
  Zoltan_Initialize(argc, argv, &version);

  // Create Zoltan object
  Zoltan zoltan;

  // Set parameters
  zoltan.Set_Param( "NUM_GID_ENTRIES", "1");  
  zoltan.Set_Param( "NUM_LID_ENTRIES", "0");  

  zoltan.Set_Param( "NUM_GLOBAL_PARTS", boost::lexical_cast<std::string>(MPI::num_processes())); 
  zoltan.Set_Param( "NUM_LOCAL_PARTS", "1"); 
  zoltan.Set_Param( "LB_METHOD", "GRAPH"); 
  zoltan.Set_Param( "LB_APPROACH", "PARTITION"); 
  //  zoltan.Set_Param( "RETURN_LISTS", "PARTS"); 
  //  zoltan.Set_Param( "EDGE_WEIGHT_DIM", "0"); 

  // Set call-back functions
  void *data_ptr = (void *)&local_graph;
  
  zoltan.Set_Num_Obj_Fn(get_number_of_objects, data_ptr);
  zoltan.Set_Obj_List_Fn(get_object_list, data_ptr);
  zoltan.Set_Num_Edges_Multi_Fn(get_number_edges, data_ptr);
  zoltan.Set_Edge_List_Multi_Fn(get_all_edges, data_ptr);

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

  std::size_t proc = MPI::process_number();
  
  if (rc != ZOLTAN_OK)
  {
    dolfin_error("ZoltanPartition.cpp",
                 "partition mesh using Zoltan",
                 "Call to Zoltan failed");
  }

  cell_partition.assign(local_graph.size(), proc);
  
  std::size_t offset = MPI::global_offset(local_graph.size(), true);
  
  for(std::size_t i = 0; i < (std::size_t)num_export; ++i)
  {
    const std::size_t idx = export_gids[i] - offset;
    cell_partition[idx] = (std::size_t)export_procs[i];
  }

  // Free data structures allocated by Zoltan::LB_Partition
  zoltan.LB_Free_Part(&import_gids, &import_lids, &import_procs, &import_parts);
  zoltan.LB_Free_Part(&export_gids, &export_lids, &export_procs, &export_parts);


}
//-----------------------------------------------------------------------------
#else
void ZoltanPartition::compute_partition(std::vector<std::size_t>& cell_partition,
                                        const LocalMeshData& mesh_data)
{
  dolfin_error("ZoltanPartition.cpp",
               "partition mesh using Zoltan",
               "DOLFIN has been configured without support for Zoltan from Trilinos");
}
#endif
//-----------------------------------------------------------------------------
