// Copyright (C) 2010 Garth N. Wells.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2010-11-15
// Last changed:

#ifdef HAS_TRILINOS

#include "dolfin/log/log.h"
#include "CellColoring.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
CellColoring::CellColoring(const Mesh& mesh) : mesh(mesh)
{
  error("CellColoring still being implemented.");
}
//-----------------------------------------------------------------------------
MeshFunction<dolfin::uint> CellColoring::compute_local_cell_coloring()
{
  // Initialise Zoltan
  float version;
  int argc = 0;
  char** argv = NULL;
  Zoltan_Initialize(argc, argv, &version);

  // Create Zoltan object
  Zoltan zoltan;

  // Set parameters
  //zoltan.Set_Param( "ORDER_METHOD", "METIS");
  zoltan.Set_Param( "ORDER_METHOD", "SCOTCH");
  zoltan.Set_Param( "NUM_GID_ENTRIES", "1");  // global ID is 1 integer
  zoltan.Set_Param( "NUM_LID_ENTRIES", "1");  // local ID is 1 integer
  zoltan.Set_Param( "OBJ_WEIGHT_DIM", "0");   // omit object weights

  // Set call-back functions
  zoltan.Set_Num_Obj_Fn(CellColoring::get_number_of_objects, this);
  zoltan.Set_Obj_List_Fn(CellColoring::get_object_list, this);
  zoltan.Set_Num_Edges_Multi_Fn(CellColoring::get_number_edges, this);
  zoltan.Set_Edge_List_Multi_Fn(CellColoring::get_all_edges, this);

  // Create array for global ids that should be renumbered
  ZOLTAN_ID_PTR  global_ids = new ZOLTAN_ID_TYPE[num_global_cells()];
  for (int i = 0; i < num_global_cells(); ++i)
    global_ids[i] = i;

  // Create array for renumbered vertices
  ZOLTAN_ID_PTR new_id = new ZOLTAN_ID_TYPE[num_global_cells()];

  // Compute re-ordering
  int rc = zoltan.Order(1, num_global_cells(), global_ids, new_id);

  // Check for errors
  if (rc != ZOLTAN_OK)
    error("Partitioning failed");

  // Copy renumber into a vector
  std::vector<uint> map(num_global_cells());
  for (uint i = 0; i < map.size(); ++i)
    map[i] = new_id[i];

  // Clean up
  delete global_ids;
  delete new_id;

  return MeshFunction<uint>(mesh);
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
  // Compute facets and facet - cell connectivity if not already computed
  const uint D = mesh.topology().dim();
  mesh.init(D - 1);
  mesh.init(D - 1, D);

  // Clear graph data
  neighbours.clear();
  neighbours.resize(mesh.num_cells());

  // Compute number of cells sharing a facet
  for (FacetIterator facet(mesh); !facet.end(); ++facet)
  {
    // Number of connected cells
    const uint num_cells = facet->num_entities(D);
    if (num_cells == 2)
    {
      // Get cell indices
      const uint cell0 = facet->entities(mesh.topology().dim())[0];
      const uint cell1 = facet->entities(mesh.topology().dim())[1];

      // Insert graph edges
      neighbours[cell0].insert(cell1);
      neighbours[cell1].insert(cell0);
    }
  }

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
          ZOLTAN_ID_PTR globalID, ZOLTAN_ID_PTR localID,
          int wgt_dim, float *obj_wgts, int *ierr)
{
  CellColoring *objs = (CellColoring *)data;
  *ierr = ZOLTAN_OK;
  for (int i = 0; i< objs->num_local_cells(); i++)
  {
    globalID[i] = i;
    localID[i] = i;
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
void CellColoring::get_all_edges(void *data, int num_gid_entries,
                              int num_lid_entries, int num_obj,
                              ZOLTAN_ID_PTR global_ids,
                              ZOLTAN_ID_PTR local_ids,
                              int *num_edges,
                              ZOLTAN_ID_PTR nbor_global_id,
                              int *nbor_procs, int wgt_dim,
                              float *ewgts, int *ierr)
{
  std::cout << "Testing:" << num_gid_entries << "  " << num_lid_entries << std::endl;

  CellColoring *objs = (CellColoring *)data;

  const std::vector<boost::unordered_set<uint> >& neighbours = objs->neighbours();

  uint sum = 0;
  for (uint i = 0; i < edges.size(); ++i)
  {
    assert(edges[i].size() == (uint) num_edges[i]);
    for (uint j = 0; j < edges[i].size(); ++j)
      nbor_global_id[sum*num_gid_entries + j] = edges[i][j];
    sum += edges[i].size();
  }
}
//-----------------------------------------------------------------------------
#endif
