// Copyright (C) 2011-2013 Garth N. Wells
//
// This file is part of DOLFINX (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include <Eigen/Dense>
#include <array>
#include <dolfinx/common/MPI.h>
#include <dolfinx/common/types.h>
#include <map>
#include <numeric>
#include <set>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

namespace dolfinx
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

  /// Compute number of cells connected to each facet (globally). Facets
  /// on internal boundaries will be connected to two cells (with the
  /// cells residing on neighboring processes)
  static void init_facet_cell_connections(Mesh& mesh);

  /// Reorder the values according to explicit global indices,
  /// distributing evenly across processes
  /// @param[in] mpi_comm MPI Communicator
  /// @param[in] values Values to reorder
  /// @param[in] global_indices Global index for each row of values
  static Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
  reorder_by_global_indices(
      MPI_Comm mpi_comm,
      const Eigen::Ref<const Eigen::Array<
          double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>& values,
      const std::vector<std::int64_t>& global_indices);

  /// Reorder the values according to explicit global indices,
  /// distributing evenly across processes
  /// @param[in] mpi_comm MPI Communicator
  /// @param[in] values Complex values to reorder
  /// @param[in] global_indices Global index for each row of values
  /// @return
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
} // namespace dolfinx
