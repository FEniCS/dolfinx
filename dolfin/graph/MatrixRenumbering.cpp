// Copyright (C) 2010 Garth N. Wells
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
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with DOLFIN.  If not, see <http://www.gnu.org/licenses/>.
//
// First added:  2010-11-15
// Last changed:

#ifdef HAS_TRILINOS

#include <boost/scoped_array.hpp>

#include "dolfin/log/log.h"
#include "dolfin/common/MPI.h"
#include "MatrixRenumbering.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
MatrixRenumbering::MatrixRenumbering(const SparsityPattern& sparsity_pattern)
      : sparsity_pattern(sparsity_pattern)
{
  //if (sparsity_pattern.rank() != 2)
  //  error("Can only create Zoltan object for SparsityPattern of rank 2.");

  if (sparsity_pattern.rank() != 2)
    error("Can only create Zoltan object for SparsityPattern of rank 2.");

  if (sparsity_pattern.size(0) != sparsity_pattern.size(1))
    error("Can only create Zoltan object square SparsityPattern (for now).");
}
//-----------------------------------------------------------------------------
std::vector<dolfin::uint> MatrixRenumbering::compute_local_renumbering_map()
{
  // Initialise Zoltan
  float version;
  int argc = 0;
  char** argv = NULL;
  Zoltan_Initialize(argc, argv, &version);

  // Create Zoltan object
  //Zoltan zoltan(MPI::COMM_WORLD);
  Zoltan zoltan;

  // Set parameters
  //zoltan.Set_Param( "ORDER_METHOD", "METIS");
  zoltan.Set_Param( "ORDER_METHOD", "SCOTCH");
  zoltan.Set_Param( "NUM_GID_ENTRIES", "1");  // global ID is 1 integer
  zoltan.Set_Param( "NUM_LID_ENTRIES", "1");  // local ID is 1 integer
  zoltan.Set_Param( "OBJ_WEIGHT_DIM", "0");   // omit object weights

  // Set call-back functions
  zoltan.Set_Num_Obj_Fn(MatrixRenumbering::get_number_of_objects, this);
  zoltan.Set_Obj_List_Fn(MatrixRenumbering::get_object_list, this);
  zoltan.Set_Num_Edges_Multi_Fn(MatrixRenumbering::get_number_edges, this);
  zoltan.Set_Edge_List_Multi_Fn(MatrixRenumbering::get_all_edges, this);

  // Create array for global ids that should be renumbered
  std::vector<ZOLTAN_ID_TYPE> global_ids(num_global_objects());
  for (uint i = 0; i < global_ids.size(); ++i)
    global_ids[i] = i;

  // Create array for renumbered vertices
  std::vector<ZOLTAN_ID_TYPE> new_id(num_global_objects());

  // Compute re-ordering
  int rc = zoltan.Order(1, num_global_objects(), &global_ids[0], &new_id[0]);

  // Check for errors
  if (rc != ZOLTAN_OK)
    error("Partitioning failed");

  // Copy renumber into a vector (in case Zoltan uses something other than uint)
  std::vector<uint> map(new_id.begin(), new_id.end());

  return map;
}
//-----------------------------------------------------------------------------
int MatrixRenumbering::num_global_objects() const
{
  return sparsity_pattern.size(0);
}
//-----------------------------------------------------------------------------
int MatrixRenumbering::num_local_objects() const
{
  return sparsity_pattern.size(0);
}
//-----------------------------------------------------------------------------
void MatrixRenumbering::num_edges_per_vertex(std::vector<uint>& num_edges) const
{
  sparsity_pattern.num_nonzeros_diagonal(num_edges);
}
//-----------------------------------------------------------------------------
const std::vector<std::vector<dolfin::uint> > MatrixRenumbering::edges() const
{
  return sparsity_pattern.diagonal_pattern(SparsityPattern::unsorted);
}
//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------
int MatrixRenumbering::get_number_of_objects(void *data, int *ierr)
{
  MatrixRenumbering *objs = (MatrixRenumbering *)data;
  *ierr = ZOLTAN_OK;
  return objs->num_local_objects();
}
//-----------------------------------------------------------------------------
void MatrixRenumbering::get_object_list(void *data, int sizeGID, int sizeLID,
          ZOLTAN_ID_PTR globalID, ZOLTAN_ID_PTR localID,
          int wgt_dim, float *obj_wgts, int *ierr)
{
  MatrixRenumbering *objs = (MatrixRenumbering *)data;
  *ierr = ZOLTAN_OK;
  for (int i = 0; i< objs->num_local_objects(); i++)
  {
    globalID[i] = i;
    localID[i] = i;
  }
}
//-----------------------------------------------------------------------------
void MatrixRenumbering::get_number_edges(void *data, int num_gid_entries,
                                       int num_lid_entries,
                                       int num_obj, ZOLTAN_ID_PTR global_ids,
                                       ZOLTAN_ID_PTR local_ids, int *num_edges,
                                       int *ierr)
{
  MatrixRenumbering *objs = (MatrixRenumbering *)data;

  //std::cout << "Testing global id entires: " << num_gid_entries << "  " << objs->num_global_objects() << std::endl;
  //std::cout << "Testing local id entires: "  << num_lid_entries << "  " << objs->num_local_objects() << std::endl;
  //assert(num_gid_entries == objs->num_global_objects());
  //assert(num_lid_entries == objs->num_local_objects());

  // Get number of edges for each graph vertex
  std::vector<uint> number_edges;
  objs->num_edges_per_vertex(number_edges);

  // Fill array num_edges array
  for (uint i = 0; i < number_edges.size(); ++i)
    num_edges[i] = number_edges[i];
}
//-----------------------------------------------------------------------------
void MatrixRenumbering::get_all_edges(void *data, int num_gid_entries,
                              int num_lid_entries, int num_obj,
                              ZOLTAN_ID_PTR global_ids,
                              ZOLTAN_ID_PTR local_ids,
                              int *num_edges,
                              ZOLTAN_ID_PTR nbor_global_id,
                              int *nbor_procs, int wgt_dim,
                              float *ewgts, int *ierr)
{
  std::cout << "Testing:" << num_gid_entries << "  " << num_lid_entries << std::endl;

  MatrixRenumbering *objs = (MatrixRenumbering *)data;
  const std::vector<std::vector<uint> > edges = objs->edges();

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
