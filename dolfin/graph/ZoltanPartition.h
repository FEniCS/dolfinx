// Copyright (C) 2013 Chris Richardson
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
// First added:  2013-02-15
// Last changed: 2013-02-26

#ifndef __DOLFIN_ZOLTAN_PARTITION_H
#define __DOLFIN_ZOLTAN_PARTITION_H

#include <vector>
#include <dolfin/common/MPI.h>

#ifdef HAS_TRILINOS
#include <zoltan_cpp.h>
#endif

namespace dolfin
{

  class LocalMeshData;

  /// This class partitions a graph using Zoltan (part of Trilinos).

  class ZoltanPartition
  {

  public:

    /// Calculate partitioning using Parallel HyperGraph (Zoltan PHG)
    static void compute_partition_phg(const MPI_Comm mpi_comm,
                                      std::vector<int>& cell_partition,
                                      const LocalMeshData& mesh_data);

    /// Calculate partitioning using recursive block bisection
    /// (Zoltan RCB - geometric partitioner)
    static void compute_partition_rcb(const MPI_Comm mpi_comm,
                                      std::vector<int>& cell_partition,
                                      const LocalMeshData& mesh_data);

  private:

    #ifdef HAS_TRILINOS

    static void num_vertex_edges(void * data, unsigned int* num_edges);

    static int get_number_of_objects(void* data, int* ierr);

    static void get_object_list(void *data,
                                int sizeGID, int sizeLID,
                                ZOLTAN_ID_PTR global_id,
                                ZOLTAN_ID_PTR local_id, int wgt_dim,
                                float* obj_wgts, int* ierr);

    static void get_number_edges(void *data,
                                 int num_gid_entries,
                                 int num_lid_entries,
                                 int num_obj, ZOLTAN_ID_PTR global_ids,
                                 ZOLTAN_ID_PTR local_ids, int *num_edges,
                                 int *ierr);

    static void get_all_edges(void* data,
                              int num_gid_entries,
                              int num_lid_entries, int num_obj,
                              ZOLTAN_ID_PTR global_ids,
                              ZOLTAN_ID_PTR local_ids,
                              int* num_edges,
                              ZOLTAN_ID_PTR nbor_global_id,
                              int* nbor_procs, int wgt_dim,
                              float* ewgts, int* ierr);


    static void get_all_geom(void *data,
                             int num_gid_entries, int num_lid_entries,
                             int num_obj,
                             ZOLTAN_ID_PTR global_ids, ZOLTAN_ID_PTR local_ids,
                             int num_dim, double *geom_vec, int *ierr);

    static int get_geom(void* data, int* ierr);

    #endif

  };
}

#endif
