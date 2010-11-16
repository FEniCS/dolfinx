// Copyright (C) 2010 Garth N. Wells.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2010-11-15
// Last changed: 2010-11-16

#ifndef __DOLFIN_ZOLTAN_CELL_COLORING_H
#define __DOLFIN_ZOLTAN_CELL_COLORING_H

#ifdef HAS_TRILINOS

#include <vector>
#include <boost/unordered_set.hpp>
#include <zoltan_cpp.h>
#include "dolfin/common/types.h"
#include <dolfin/mesh/Cell.h>

namespace dolfin
{

  class Mesh;

  /// This class computes cell colorings for a local mesh. It supports vertex,
  /// facet and edge-based colorings.  Zoltan (part of Trilinos) is used to
  /// the colorings.

  class CellColoring
  {

  public:

    /// Constructor
    CellColoring(const Mesh& mesh, std::string type="vertex");

    /// Compute cell colors
    CellFunction<uint> compute_local_cell_coloring();

  private:

    // Build graph that is to be colored
    template<class T> void build_graph();

    /// Number of global cells (graph vertices)
    int num_global_cells() const;

    /// Number of local cells (graph vertices)
    int num_local_cells() const;

    /// Number of neighboring cells
    void num_neighbors(uint* num_neighbors) const;

    // Mesh
    const Mesh& mesh;

    // Graph (cell neighbours)
    std::vector<boost::unordered_set<uint> > neighbours;

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

}

#endif
#endif
