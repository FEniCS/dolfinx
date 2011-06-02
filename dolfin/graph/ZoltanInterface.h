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
// First added:  2010-11-16
// Last changed:

#ifndef __DOLFIN_ZOLTAN_GRAPH_COLORING_H
#define __DOLFIN_ZOLTAN_GRAPH_COLORING_H

#ifdef HAS_TRILINOS
#include <zoltan_cpp.h>
#endif

#include <dolfin/common/Array.h>
#include <dolfin/common/types.h>
#include "Graph.h"

namespace dolfin
{

  class Mesh;

  /// This class colors a graph using Zoltan (part of Trilinos). It is designed
  /// to work on a single process.

  class ZoltanInterface
  {

  public:

    /// Compute vertex colors
    static uint compute_local_vertex_coloring(const Graph& graph, Array<uint>& colors);

  private:

    #ifdef HAS_TRILINOS
    class ZoltanGraphInterface
    {

      public:

      /// Constructor
      ZoltanGraphInterface(const Graph& graph);

      /// Graph object
      const Graph& graph;

      /// Number of edges from each vertex
      void num_vertex_edges(uint* num_edges) const;

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

    };
    #endif

  };
}

#endif
