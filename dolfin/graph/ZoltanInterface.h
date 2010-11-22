// Copyright (C) 2010 Garth N. Wells.
// Licensed under the GNU LGPL Version 2.1.
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
    static void compute_local_vertex_coloring(const Graph& graph, Array<uint>& colors);

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
