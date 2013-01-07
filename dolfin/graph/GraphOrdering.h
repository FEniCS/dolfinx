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
// First added:  2010-11-15
// Last changed:

#ifndef __DOLFIN_ZOLTAN_ORDERING_H
#define __DOLFIN_ZOLTAN_ORDERING_H

#ifdef HAS_TRILINOS

#include <cstddef>
#include <vector>
#include <zoltan_cpp.h>
#include "dolfin/common/Set.h"

namespace dolfin
{


  class TensorLayout;

  /// This class computes re-ordering based on a SparsityPattern graph
  /// representation of a sparse matrix. It uses Zoltan, which is part of
  /// Trilinos.

  class GraphOrdering
  {

  public:

    /// Constructor
    GraphOrdering(const TensorLayout& tensor_layout);

    /// Compute re-ordering for process (map[old] -> new)
    std::vector<std::size_t> compute_local_reordering_map();

  private:

    /// Number of global graph vertices
    int num_global_objects() const;

    /// Number of local graph vertices
    int num_local_objects() const;

    /// Number of edges per vertex
    void num_edges_per_vertex(std::vector<std::size_t>& num_edges) const;

    /// Vertex edges
    const std::vector<std::vector<std::size_t> > edges() const;

    // Zoltan call-back functions

    static int get_number_of_objects(void* data, int* ierr);

    static void get_object_list(void* data, int sizeGID, int sizeLID,
                                ZOLTAN_ID_PTR globalID,
                                ZOLTAN_ID_PTR localID, int wgt_dim,
                                float* obj_wgts, int* ierr);

    static void get_number_edges(void* data, int num_gid_entries,
                                 int num_lid_entries,
                                 int num_obj, ZOLTAN_ID_PTR global_ids,
                                 ZOLTAN_ID_PTR local_ids, int* num_edges,
                                 int* ierr);

    static void get_all_edges(void* data, int num_gid_entries,
                              int num_lid_entries, int num_obj,
                              ZOLTAN_ID_PTR global_ids,
                              ZOLTAN_ID_PTR local_ids,
                              int* num_edges,
                              ZOLTAN_ID_PTR nbor_global_id,
                              int* nbor_procs, int wgt_dim,
                              float* ewgts, int* ierr);


    // Tensor layout
    const TensorLayout& tensor_layout;

  };

}

#endif
#endif
