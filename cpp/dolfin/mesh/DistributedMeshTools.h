// Copyright (C) 2011-2013 Garth N. Wells
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include <array>
#include <dolfin/common/MPI.h>
#include <dolfin/common/types.h>
#include <map>
#include <numeric>
#include <set>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

namespace dolfin
{

namespace mesh
{
class Mesh;

/// This class provides various functionality for working with
/// distributed meshes.

class DistributedMeshTools
{
public:
  /// Create global entity indices for entities of dimension d
  static void number_entities(const Mesh& mesh, int d);

  /// Create global entity indices for entities of dimension d for given
  /// global vertex indices. Returns  global_entity_indices,
  /// shared_entities, and XXXX?
  static std::tuple<std::vector<std::int64_t>,
                    std::map<std::int32_t, std::set<std::int32_t>>, std::size_t>
  number_entities_computation(
      const Mesh& mesh,
      int d);

  /// Compute number of cells connected to each facet (globally). Facets
  /// on internal boundaries will be connected to two cells (with the
  /// cells residing on neighboring processes)
  static void init_facet_cell_connections(Mesh& mesh);

  /// Find processes that own or share mesh entities (using entity
  /// global indices). Returns (global_dof, set(process_num,
  /// local_index)). Exclusively local entities will not appear in the
  /// map. Works only for vertices and cells
  static std::map<std::size_t, std::set<std::pair<std::size_t, std::size_t>>>
  locate_off_process_entities(const std::vector<std::size_t>& entity_indices,
                              std::size_t dim, const Mesh& mesh);

  /// Compute map from local index of shared entity to list of sharing
  /// process and local index, i.e. (local index, [(sharing process p,
  /// local index on p)])
  static std::unordered_map<std::int32_t,
                            std::vector<std::pair<std::int32_t, std::int32_t>>>
  compute_shared_entities(const Mesh& mesh, std::size_t d);

  /// Reorder the values according to explicit global indices, distributing
  /// evenly across processes
  /// @param mpi_comm
  ///    MPI Communicator
  /// @param values
  ///    Values to reorder
  /// @param global_indices
  ///    Global index for each row of values
  static Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
  reorder_by_global_indices(
      MPI_Comm mpi_comm,
      const Eigen::Ref<const Eigen::Array<
          double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>& values,
      const std::vector<std::int64_t>& global_indices);
  /// Reorder the values according to explicit global indices, distributing
  /// evenly across processes
  /// @param mpi_comm
  ///    MPI Communicator
  /// @param values
  ///    Complex values to reorder
  /// @param global_indices
  ///    Global index for each row of values
  static Eigen::Array<std::complex<double>, Eigen::Dynamic, Eigen::Dynamic,
                      Eigen::RowMajor>
  reorder_by_global_indices(
      MPI_Comm mpi_comm,
      const Eigen::Ref<const Eigen::Array<std::complex<double>, Eigen::Dynamic,
                                          Eigen::Dynamic, Eigen::RowMajor>>&
          values,
      const std::vector<std::int64_t>& global_indices);
};
} // namespace mesh
} // namespace dolfin
