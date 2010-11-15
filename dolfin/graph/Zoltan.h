// Copyright (C) 2010 Garth N. Wells.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2010-11-15
// Last changed:

#ifndef __DOLFIN_ZOLTAN_H
#define __DOLFIN_ZOLTAN_H

#ifdef HAS_TRILINOS

#include <vector>
#include <zoltan_cpp.h>
#include "dolfin/common/Set.h"
#include "dolfin/common/types.h"
#include <dolfin/la/SparsityPattern.h>

namespace dolfin
{

  /// This class provides an interface for Zoltan

  class ZoltanInterface
  {

  public:

    /// Create Zoltan object for a graph defined by a sparsity pattern
    ZoltanInterface(const SparsityPattern& sparsity_pattern);

  private:

    /// Number of global graph vertices
    int num_global_objects() const;

    /// Number of local graph vertices
    int num_local_objects() const;

    /// Number of edges per vertex
    void num_edges_per_vertex(uint* num_edges) const;

    /// Vertex edges
    const std::vector<Set<uint> >& edges() const;

  public:

    /// Runumbering map on on process (map[old] -> new)
    std::vector<uint> local_renumbering_map();

  private:

    // Zoltan call-back functions

    static int get_number_of_objects(void *data, int *ierr);

    static void get_object_list(void *data, int sizeGID, int sizeLID,
                                ZOLTAN_ID_PTR globalID,
                                ZOLTAN_ID_PTR localID, int wgt_dim,
                                float *obj_wgts, int *ierr);

    static void get_number_edges(void *data, int num_gid_entries,
                                 int num_lid_entries,
                                 int num_obj, ZOLTAN_ID_PTR global_ids,
                                 ZOLTAN_ID_PTR local_ids, int *num_edges,
                                 int *ierr);

    static void get_all_edges(void *data, int num_gid_entries,
                              int num_lid_entries, int num_obj,
                              ZOLTAN_ID_PTR global_ids,
                              ZOLTAN_ID_PTR local_ids,
                              int *num_edges,
                              ZOLTAN_ID_PTR nbor_global_id,
                              int *nbor_procs, int wgt_dim,
                              float *ewgts, int *ierr);


    const SparsityPattern& sparsity_pattern;

  };

}

#endif
#endif
