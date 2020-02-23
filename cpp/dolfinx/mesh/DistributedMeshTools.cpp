// Copyright (C) 2011-2019 Garth N. Wells and Chris Richardson
//
// This file is part of DOLFINX (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "DistributedMeshTools.h"
#include "MeshFunction.h"
#include "cell_types.h"
#include "dolfinx/common/IndexMap.h"
#include "dolfinx/common/MPI.h"
#include "dolfinx/common/Timer.h"
#include "dolfinx/graph/Graph.h"
#include "dolfinx/graph/SCOTCH.h"
#include <Eigen/Dense>
#include <complex>
#include <dolfinx/common/log.h>

using namespace dolfinx;
using namespace dolfinx::mesh;

//-----------------------------------------------------------------------------
namespace
{
template <typename T>
Eigen::Array<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
reorder_values_by_global_indices(
    MPI_Comm mpi_comm,
    const Eigen::Ref<const Eigen::Array<T, Eigen::Dynamic, Eigen::Dynamic,
                                        Eigen::RowMajor>>& values,
    const std::vector<std::int64_t>& global_indices)
{
  dolfinx::common::Timer t("DistributedMeshTools: reorder values");

  // Number of items to redistribute
  const std::size_t num_local_indices = global_indices.size();
  assert(num_local_indices == (std::size_t)values.rows());

  // Calculate size of overall global vector by finding max index value
  // anywhere
  const std::size_t global_vector_size
      = dolfinx::MPI::max(mpi_comm, *std::max_element(global_indices.begin(),
                                                      global_indices.end()))
        + 1;

  // Send unwanted values off process
  const std::size_t mpi_size = dolfinx::MPI::size(mpi_comm);
  std::vector<std::vector<std::size_t>> indices_to_send(mpi_size);
  std::vector<std::vector<T>> values_to_send(mpi_size);

  // Go through local vector and append value to the appropriate list to
  // send to correct process
  for (std::size_t i = 0; i != num_local_indices; ++i)
  {
    const std::size_t global_i = global_indices[i];
    const std::size_t process_i
        = dolfinx::MPI::index_owner(mpi_comm, global_i, global_vector_size);
    indices_to_send[process_i].push_back(global_i);
    values_to_send[process_i].insert(values_to_send[process_i].end(),
                                     values.row(i).data(),
                                     values.row(i).data() + values.cols());
  }

  // Redistribute the values to the appropriate process - including
  // self. All values are "in the air" at this point. Receive into flat
  // arrays.
  std::vector<std::size_t> received_indices;
  std::vector<T> received_values;
  dolfinx::MPI::all_to_all(mpi_comm, indices_to_send, received_indices);
  dolfinx::MPI::all_to_all(mpi_comm, values_to_send, received_values);

  // Map over received values as Eigen array
  assert(received_indices.size() * values.cols() == received_values.size());
  Eigen::Map<Eigen::Array<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
      received_values_array(received_values.data(), received_indices.size(),
                            values.cols());

  // Create array for new data. Note that any indices which are not
  // received will be uninitialised.
  const std::array<std::int64_t, 2> range
      = dolfinx::MPI::local_range(mpi_comm, global_vector_size);
  Eigen::Array<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> new_values(
      range[1] - range[0], values.cols());

  // Go through received data in descending order, and place in local
  // partition of the global vector. Any duplicate data (with same
  // index) will be overwritten by values from the lowest rank process.
  for (std::int32_t j = received_indices.size() - 1; j >= 0; --j)
  {
    const std::int64_t global_i = received_indices[j];
    assert(global_i >= range[0] && global_i < range[1]);
    new_values.row(global_i - range[0]) = received_values_array.row(j);
  }

  return new_values;
}
//-----------------------------------------------------------------------------
} // namespace

//-----------------------------------------------------------------------------
Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
DistributedMeshTools::reorder_by_global_indices(
    MPI_Comm mpi_comm,
    const Eigen::Ref<const Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic,
                                        Eigen::RowMajor>>& values,
    const std::vector<std::int64_t>& global_indices)
{
  return reorder_values_by_global_indices<double>(mpi_comm, values,
                                                  global_indices);
}
//-----------------------------------------------------------------------------
Eigen::Array<std::complex<double>, Eigen::Dynamic, Eigen::Dynamic,
             Eigen::RowMajor>
DistributedMeshTools::reorder_by_global_indices(
    MPI_Comm mpi_comm,
    const Eigen::Ref<const Eigen::Array<std::complex<double>, Eigen::Dynamic,
                                        Eigen::Dynamic, Eigen::RowMajor>>&
        values,
    const std::vector<std::int64_t>& global_indices)
{
  return reorder_values_by_global_indices<std::complex<double>>(
      mpi_comm, values, global_indices);
}
//-----------------------------------------------------------------------------
