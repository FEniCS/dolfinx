// Copyright (C) 2010 Garth N. Wells.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2010-11-15
// Last changed: 2010-11-16

#ifdef HAS_TRILINOS

#include <boost/foreach.hpp>

#include "dolfin/log/log.h"
#include "dolfin/common/Timer.h"
#include "dolfin/mesh/Cell.h"
#include "dolfin/mesh/Edge.h"
#include "dolfin/mesh/Facet.h"
#include "dolfin/mesh/Mesh.h"
#include "dolfin/mesh/Vertex.h"
#include "CellColoring.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
CellColoring::CellColoring(const Mesh& mesh, std::string type) : mesh(mesh)
{
  if (type != "vertex" && type != "facet" && type != "edge")
    error("Coloring type unkown. Options are \"vertex\", \"facet\" or \"edge\".");

  // Resize graph data
  neighbours.resize(mesh.num_cells());

  // Build graph
  if (type == "vertex")
    build_graph<VertexIterator>();
  else if (type == "facet")
  {
    // Compute facets and facet - cell connectivity if not already computed
    const uint D = mesh.topology().dim();
    mesh.init(D - 1);
    mesh.init(D - 1, D);

    build_graph<FacetIterator>();
  }
  else if (type == "edge")
  {
    // Compute edges and edges - cell connectivity if not already computed
    const uint D = mesh.topology().dim();
    mesh.init(1);
    mesh.init(1, D);

    build_graph<EdgeIterator>();
  }
}
//-----------------------------------------------------------------------------
CellFunction<dolfin::uint> CellColoring::compute_local_cell_coloring()
{
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
  zoltan.Set_Num_Obj_Fn(CellColoring::get_number_of_objects, this);
  zoltan.Set_Obj_List_Fn(CellColoring::get_object_list, this);
  zoltan.Set_Num_Edges_Multi_Fn(CellColoring::get_number_edges, this);
  zoltan.Set_Edge_List_Multi_Fn(CellColoring::get_all_edges, this);

  // Create array for global ids that should be renumbered
  ZOLTAN_ID_PTR  global_ids = new ZOLTAN_ID_TYPE[num_global_cells()];
  for (int i = 0; i < num_global_cells(); ++i)
    global_ids[i] = i;

  // Create array to hold colours
  CellFunction<uint> colors(mesh);

  // Call Zoltan function to compute coloring
  int tmp = 1;
  int rc = zoltan.Color(tmp, num_global_cells(), global_ids, reinterpret_cast<int*>(colors.values()));
  if (rc != ZOLTAN_OK)
    error("Partitioning failed");

  // Clean up
  delete global_ids;

  return colors;
}
//-----------------------------------------------------------------------------
template<class T> void CellColoring::build_graph()
{
  for (CellIterator cell(mesh); !cell.end(); ++cell)
  {
    const uint cell_index = cell->index();
    for (T entity(*cell); !entity.end(); ++entity)
    {
      for (CellIterator ncell(*entity); !ncell.end(); ++ncell)
        neighbours[cell_index].insert(ncell->index());
    }
  }
}
//-----------------------------------------------------------------------------
int CellColoring::num_global_cells() const
{
  return mesh.num_cells();
}
//-----------------------------------------------------------------------------
int CellColoring::num_local_cells() const
{
  return mesh.num_cells();
}
//-----------------------------------------------------------------------------
void CellColoring::num_neighbors(uint* num_neighbors) const
{
  // Compute nunber of neighbors for each cell
  for (uint i = 0; i < neighbours.size(); ++i)
    num_neighbors[i] = neighbours[i].size();
}
//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------
int CellColoring::get_number_of_objects(void* data, int* ierr)
{
  CellColoring *objs = (CellColoring *)data;
  *ierr = ZOLTAN_OK;
  return objs->num_local_cells();
}
//-----------------------------------------------------------------------------
void CellColoring::get_object_list(void *data, int sizeGID, int sizeLID,
                                   ZOLTAN_ID_PTR global_id,
                                   ZOLTAN_ID_PTR local_id, int wgt_dim,
                                   float* obj_wgts, int* ierr)
{
  CellColoring *objs = (CellColoring *)data;
  *ierr = ZOLTAN_OK;
  for (int i = 0; i< objs->num_local_cells(); i++)
  {
    global_id[i] = i;
    local_id[i] = i;
  }
}
//-----------------------------------------------------------------------------
void CellColoring::get_number_edges(void *data, int num_gid_entries,
                                    int num_lid_entries,
                                    int num_obj, ZOLTAN_ID_PTR global_ids,
                                    ZOLTAN_ID_PTR local_ids, int *num_edges,
                                    int *ierr)
{
  CellColoring *objs = (CellColoring *)data;
  objs->num_neighbors(reinterpret_cast<uint*>(num_edges));
}
//-----------------------------------------------------------------------------
void CellColoring::get_all_edges(void* data, int num_gid_entries,
                                 int num_lid_entries, int num_obj,
                                 ZOLTAN_ID_PTR global_ids,
                                 ZOLTAN_ID_PTR local_ids,
                                 int* num_edges,
                                 ZOLTAN_ID_PTR nbor_global_id,
                                 int* nbor_procs, int wgt_dim,
                                 float* ewgts, int* ierr)
{
  CellColoring *objs = (CellColoring *)data;

  // Get graph
  const std::vector<boost::unordered_set<uint> >& neighbours = objs->neighbours;

  uint entry = 0;
  for (uint i = 0; i < neighbours.size(); ++i)
  {
    assert(neighbours[i].size() == (uint) num_edges[i]);
    BOOST_FOREACH(boost::unordered_set<uint>::value_type neighbour, neighbours[i])
    {
      nbor_global_id[entry++] = neighbour;
    }
  }
}
//-----------------------------------------------------------------------------
#endif
