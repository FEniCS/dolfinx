// Copyright (C) 2010 Garth N. Wells.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2010-11-16
// Last changed:

#ifdef HAS_TRILINOS

#include <boost/foreach.hpp>
#include "dolfin/log/log.h"
#include "dolfin/common/Timer.h"
#include "ZoltanGraphColoring.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
ZoltanGraphColoring::ZoltanGraphColoring(const Graph& graph) : graph(graph)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
void ZoltanGraphColoring::compute_local_vertex_coloring(Array<uint>& colors)
{
  if (colors.size() != graph.size())
    error("ZoltanGraphColoring::compute_local_cell_coloring: colors array is of wrong length.");

  // Initialise Zoltan
  float version;
  int argc = 0;
  char** argv = NULL;
  Zoltan_Initialize(argc, argv, &version);

  // Create Zoltan object
  Zoltan zoltan;

  // Set parameters
  zoltan.Set_Param( "NUM_GID_ENTRIES", "1");  // global ID is single integer
  zoltan.Set_Param( "NUM_LID_ENTRIES", "1");  // local ID is single integer

  // Set call-back functions
  zoltan.Set_Num_Obj_Fn(ZoltanGraphColoring::get_number_of_objects, this);
  zoltan.Set_Obj_List_Fn(ZoltanGraphColoring::get_object_list, this);
  zoltan.Set_Num_Edges_Multi_Fn(ZoltanGraphColoring::get_number_edges, this);
  zoltan.Set_Edge_List_Multi_Fn(ZoltanGraphColoring::get_all_edges, this);

  // Create array for global ids that should be renumbered
  ZOLTAN_ID_PTR  global_ids = new ZOLTAN_ID_TYPE[num_global_vertices()];
  for (int i = 0; i < num_global_vertices(); ++i)
    global_ids[i] = i;

  // Call Zoltan function to compute coloring
  int num_id = 1;
  int rc = zoltan.Color(num_id, num_global_vertices(), global_ids, reinterpret_cast<int*>(colors.data().get()));
  if (rc != ZOLTAN_OK)
    error("Zoltan coloring failed");

  // Clean up
  delete global_ids;
}
//-----------------------------------------------------------------------------
int ZoltanGraphColoring::num_global_vertices() const
{
  return graph.size();
}
//-----------------------------------------------------------------------------
int ZoltanGraphColoring::num_local_vertices() const
{
  return graph.size();
}
//-----------------------------------------------------------------------------
void ZoltanGraphColoring::num_vertex_edges(uint* num_edges) const
{
  assert(num_edges);

  // Compute nunber of edges from each graph node
  for (uint i = 0; i < graph.size(); ++i)
    num_edges[i] = graph[i].size();
}
//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------
int ZoltanGraphColoring::get_number_of_objects(void* data, int* ierr)
{
  ZoltanGraphColoring *objs = (ZoltanGraphColoring *)data;
  *ierr = ZOLTAN_OK;
  return objs->num_local_vertices();
}
//-----------------------------------------------------------------------------
void ZoltanGraphColoring::get_object_list(void *data, int sizeGID, int sizeLID,
                                   ZOLTAN_ID_PTR global_id,
                                   ZOLTAN_ID_PTR local_id, int wgt_dim,
                                   float* obj_wgts, int* ierr)
{
  ZoltanGraphColoring *objs = (ZoltanGraphColoring *)data;
  *ierr = ZOLTAN_OK;
  for (int i = 0; i< objs->num_local_vertices(); i++)
  {
    global_id[i] = i;
    local_id[i] = i;
  }
}
//-----------------------------------------------------------------------------
void ZoltanGraphColoring::get_number_edges(void *data, int num_gid_entries,
                                    int num_lid_entries,
                                    int num_obj, ZOLTAN_ID_PTR global_ids,
                                    ZOLTAN_ID_PTR local_ids, int *num_edges,
                                    int *ierr)
{
  ZoltanGraphColoring *objs = (ZoltanGraphColoring *)data;
  objs->num_vertex_edges(reinterpret_cast<uint*>(num_edges));
}
//-----------------------------------------------------------------------------
void ZoltanGraphColoring::get_all_edges(void* data, int num_gid_entries,
                                 int num_lid_entries, int num_obj,
                                 ZOLTAN_ID_PTR global_ids,
                                 ZOLTAN_ID_PTR local_ids,
                                 int* num_edges,
                                 ZOLTAN_ID_PTR nbor_global_id,
                                 int* nbor_procs, int wgt_dim,
                                 float* ewgts, int* ierr)
{
  ZoltanGraphColoring *objs = (ZoltanGraphColoring *)data;

  // Get graph
  const Graph graph = objs->graph;

  uint entry = 0;
  for (uint i = 0; i < graph.size(); ++i)
  {
    assert(graph[i].size() == (uint) num_edges[i]);
    BOOST_FOREACH(boost::unordered_set<uint>::value_type edge, graph[i])
    {
      nbor_global_id[entry++] = edge;
    }
  }
}
//-----------------------------------------------------------------------------
#endif
