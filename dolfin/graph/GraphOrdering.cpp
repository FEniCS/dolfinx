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
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with DOLFIN. If not, see <http://www.gnu.org/licenses/>.
//
// Modified by Anders Logg 2011
//
// First added:  2010-11-15
// Last changed: 2011-11-15

#ifdef HAS_TRILINOS

// Included here to avoid a C++ problem with some MPI implementations
#include <dolfin/common/MPI.h>

#include <zoltan_cpp.h>

#include "dolfin/common/MPI.h"
#include "dolfin/la/SparsityPattern.h"
#include "dolfin/la/TensorLayout.h"
#include "dolfin/log/log.h"
#include "GraphOrdering.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
GraphOrdering::GraphOrdering(const TensorLayout& tensor_layout)
      : _tensor_layout(tensor_layout)
{
  if (tensor_layout.rank() != 2)
  {
    dolfin_error("GraphOrdering.cpp",
                 "create matrix re-ordering",
                 "Zoltan object for sparsity pattern re-ordering can only be used for rank 2 tensors");
  }

  if (!tensor_layout.sparsity_pattern())
  {
    dolfin_error("GraphOrdering.cpp",
                 "create matrix re-ordering",
                 "TensorLayout object must a have sparsity pattern");
  }

  if (tensor_layout.size(0) != tensor_layout.size(1))
  {
    dolfin_error("GraphOrdering.cpp",
                 "create matrix re-ordering",
                 "Zoltan object for sparsity pattern re-ordering can only be used for square matrices");
  }
}
//-----------------------------------------------------------------------------
std::vector<std::size_t> GraphOrdering::compute_local_reordering_map()
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
  zoltan.Set_Num_Obj_Fn(GraphOrdering::get_number_of_objects, this);
  zoltan.Set_Obj_List_Fn(GraphOrdering::get_object_list, this);
  zoltan.Set_Num_Edges_Multi_Fn(GraphOrdering::get_number_edges, this);
  zoltan.Set_Edge_List_Multi_Fn(GraphOrdering::get_all_edges, this);

  // Create array for global ids that should be re-ordered
  std::vector<ZOLTAN_ID_TYPE> global_ids(num_global_objects());
  for (std::size_t i = 0; i < global_ids.size(); ++i)
    global_ids[i] = i;

  // Create array for re-ordered vertices
  std::vector<ZOLTAN_ID_TYPE> new_id(num_global_objects());

  // Compute re-ordering
  int rc = zoltan.Order(1, num_global_objects(), &global_ids[0], &new_id[0]);

  // Check for errors
  if (rc != ZOLTAN_OK)
  {
    dolfin_error("GraphOrdering.cpp",
                 "compute matrix re-ordering",
                 "Zoltan partitioning failed");
  }

  // Copy re-ordering into a vector (in case Zoltan uses something other than std::size_t)
  std::vector<std::size_t> map(new_id.begin(), new_id.end());

  return map;
}
//-----------------------------------------------------------------------------
int GraphOrdering::num_global_objects() const
{
  return _tensor_layout.size(0);
}
//-----------------------------------------------------------------------------
int GraphOrdering::num_local_objects() const
{
  return _tensor_layout.size(0);
}
//-----------------------------------------------------------------------------
void GraphOrdering::num_edges_per_vertex(std::vector<std::size_t>& num_edges) const
{
  dolfin_assert(_tensor_layout.sparsity_pattern());
  _tensor_layout.sparsity_pattern()->num_nonzeros_diagonal(num_edges);
}
//-----------------------------------------------------------------------------
const std::vector<std::vector<std::size_t>> GraphOrdering::edges() const
{
  dolfin_assert(_tensor_layout.sparsity_pattern());
  return _tensor_layout.sparsity_pattern()->diagonal_pattern(SparsityPattern::Type::unsorted);
}
//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------
int GraphOrdering::get_number_of_objects(void *data, int *ierr)
{
  GraphOrdering *objs = (GraphOrdering *)data;
  *ierr = ZOLTAN_OK;
  return objs->num_local_objects();
}
//-----------------------------------------------------------------------------
void GraphOrdering::get_object_list(void *data, int sizeGID, int sizeLID,
                                ZOLTAN_ID_PTR globalID, ZOLTAN_ID_PTR localID,
                                int wgt_dim, float *obj_wgts, int *ierr)
{
  GraphOrdering *objs = (GraphOrdering *)data;
  *ierr = ZOLTAN_OK;
  for (int i = 0; i< objs->num_local_objects(); i++)
  {
    globalID[i] = i;
    localID[i] = i;
  }
}
//-----------------------------------------------------------------------------
void GraphOrdering::get_number_edges(void *data, int num_gid_entries,
                                       int num_lid_entries,
                                       int num_obj, ZOLTAN_ID_PTR global_ids,
                                       ZOLTAN_ID_PTR local_ids, int *num_edges,
                                       int *ierr)
{
  GraphOrdering *objs = (GraphOrdering *)data;

  // Get number of edges for each graph vertex
  std::vector<std::size_t> number_edges;
  objs->num_edges_per_vertex(number_edges);

  // Fill array num_edges array
  for (std::size_t i = 0; i < number_edges.size(); ++i)
    num_edges[i] = number_edges[i];
}
//-----------------------------------------------------------------------------
void GraphOrdering::get_all_edges(void *data, int num_gid_entries,
                                  int num_lid_entries, int num_obj,
                                  ZOLTAN_ID_PTR global_ids,
                                  ZOLTAN_ID_PTR local_ids,
                                  int *num_edges,
                                  ZOLTAN_ID_PTR nbor_global_id,
                                  int *nbor_procs, int wgt_dim,
                                  float *ewgts, int *ierr)
{
  GraphOrdering *objs = (GraphOrdering *)data;
  const std::vector<std::vector<std::size_t>> edges = objs->edges();

  std::size_t sum = 0;
  for (std::size_t i = 0; i < edges.size(); ++i)
  {
    dolfin_assert(static_cast<int>(edges[i].size()) == num_edges[i]);
    for (std::size_t j = 0; j < edges[i].size(); ++j)
      nbor_global_id[sum*num_gid_entries + j] = edges[i][j];
    sum += edges[i].size();
  }
}
//-----------------------------------------------------------------------------

#endif
