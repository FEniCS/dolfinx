// Copyright (C) 2010 Garth N. Wells.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2010-11-15
// Last changed:

#ifndef __DOLFIN_ZOLTAN_CELL_COLORING_H
#define __DOLFIN_ZOLTAN_CELL_COLORING_H

#ifdef HAS_TRILINOS

#include <vector>
#include <boost/unordered_set.hpp>
#include <zoltan_cpp.h>
#include "dolfin/common/types.h"
#include <dolfin/mesh/Mesh.h>
#include <dolfin/mesh/MeshFunction.h>

namespace dolfin
{

  /// This class computes cell colorings for a mesh. It uses Zoltan, which is
  /// part of Trilinos.

  class CellColoring
  {

  public:

    /// Constructor
    CellColoring(const Mesh& mesh);

    /// Compute cell colors
    MeshFunction<uint> compute_local_cell_coloring();

  private:

    /// Number of global cells (graph vertices)
    int num_global_cells() const;

    /// Number of local cells (graph vertices)
    int num_local_cells() const;

    /// Number of neighboring cells
    void num_neighbors(uint* num_neighbors) const;


    // Zoltan call-back functborsions

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


    // Mesh
    const Mesh& mesh;

    // Cell neighbours
    std::vector<boost::unordered_set<uint> > neighbours;

  };

}

#endif
#endif
