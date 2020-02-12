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
void DistributedMeshTools::init_facet_cell_connections(MPI_Comm comm,
                                                       Topology& topology)
{
  // Topological dimension
  const int D = topology.dim();

  if (!topology.connectivity(D - 1, 0))
    throw std::runtime_error("Facet entities have not been computed.");
  if (!topology.connectivity(D - 1, D))
    throw std::runtime_error("Facet-cell connectivity has not been computed.");

  // Calculate the number of global cells attached to each facet
  // essentially defining the exterior surface
  // FIXME: should this be done earlier, e.g. at partitioning stage
  // when dual graph is built?

  // Create vector to hold number of cells connected to each
  // facet. Initially copy over from local values.

  assert(topology.connectivity(D - 1, 0));
  const int num_facets = topology.connectivity(D - 1, 0)->num_nodes();
  Eigen::Array<std::int32_t, Eigen::Dynamic, 1> num_global_neighbors(
      num_facets);

  const std::map<std::int32_t, std::set<std::int32_t>>& shared_facets
      = topology.shared_entities(D - 1);

  // Check if no ghost cells
  assert(topology.index_map(D));
  if (topology.index_map(D)->num_ghosts() == 0)
  {
    // Copy local values
    assert(topology.connectivity(D - 1, D));
    auto connectivity = topology.connectivity(D - 1, D);
    for (int f = 0; f < num_facets; ++f)
      num_global_neighbors[f] = connectivity->num_links(f);

    // All shared facets must have two cells, if no ghost cells
    for (const auto& f_it : shared_facets)
      num_global_neighbors[f_it.first] = 2;
  }
  else
  {
    // With ghost cells, shared facets may be on an external edge, so
    // need to check connectivity with the cell owner.

    const std::int32_t mpi_size = MPI::size(comm);
    std::vector<std::vector<std::size_t>> send_facet(mpi_size);
    std::vector<std::vector<std::size_t>> recv_facet(mpi_size);

    // Map shared facets
    std::map<std::size_t, std::size_t> global_to_local_facet;

    const Eigen::Array<int, Eigen::Dynamic, 1>& cell_owners
        = topology.index_map(D)->ghost_owners();
    const std::int32_t ghost_offset_c = topology.index_map(D)->size_local();
    const std::int32_t ghost_offset_f = topology.index_map(D - 1)->size_local();
    const std::map<std::int32_t, std::set<std::int32_t>>& sharing_map_f
        = topology.shared_entities(D - 1);
    const auto& global_facets = topology.global_indices(D - 1);
    assert(topology.connectivity(D - 1, D));
    auto connectivity = topology.connectivity(D - 1, D);
    for (int f = 0; f < num_facets; ++f)
    {
      // Insert shared facets into mapping
      if (sharing_map_f.find(f) != sharing_map_f.end())
        global_to_local_facet.insert({global_facets[f], f});

      // Copy local values
      const int n_cells = connectivity->num_links(f);
      num_global_neighbors[f] = n_cells;

      if ((f >= ghost_offset_f) and n_cells == 1)
      {
        // Singly attached ghost facet - check with owner of attached
        // cell
        auto c = connectivity->links(f);
        if (c[0] >= ghost_offset_c)
        {
          const int owner = cell_owners[c[0] - ghost_offset_c];
          send_facet[owner].push_back(global_facets[f]);
        }
      }
    }

    MPI::all_to_all(comm, send_facet, recv_facet);

    // Convert received global facet index into number of attached cells
    // and return to sender
    std::vector<std::vector<std::size_t>> send_response(mpi_size);
    for (std::int32_t p = 0; p < mpi_size; ++p)
    {
      for (auto r = recv_facet[p].begin(); r != recv_facet[p].end(); ++r)
      {
        auto map_it = global_to_local_facet.find(*r);
        assert(map_it != global_to_local_facet.end());
        const int n_cells = connectivity->num_links(map_it->second);
        send_response[p].push_back(n_cells);
      }
    }

    MPI::all_to_all(comm, send_response, recv_facet);

    // Insert received result into same facet that it came from
    for (std::int32_t p = 0; p < mpi_size; ++p)
    {
      for (std::size_t i = 0; i < recv_facet[p].size(); ++i)
      {
        auto f_it = global_to_local_facet.find(send_facet[p][i]);
        assert(f_it != global_to_local_facet.end());
        num_global_neighbors[f_it->second] = recv_facet[p][i];
      }
    }
  }

  assert(topology.connectivity(D - 1, D));
  topology.set_global_size({D - 1, D}, num_global_neighbors);
}
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
