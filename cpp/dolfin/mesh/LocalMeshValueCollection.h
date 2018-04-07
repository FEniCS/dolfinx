// Copyright (C) 2008-2012 Garth N. Wells
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include <dolfin/common/MPI.h>
#include <dolfin/log/log.h>
#include <map>
#include <utility>
#include <vector>

namespace dolfin
{

namespace mesh
{
template <typename T>
class MeshValueCollection;

/// This class stores mesh data on a local processor corresponding
/// to a portion of a MeshValueCollection.

template <typename T>
class LocalMeshValueCollection
{
public:
  /// Create local mesh data for given LocalMeshValueCollection
  LocalMeshValueCollection(MPI_Comm comm, const MeshValueCollection<T>& values,
                           std::size_t dim);

  /// Move constructor
  LocalMeshValueCollection(LocalMeshValueCollection&& lmvc) = default;

  /// Destructor
  ~LocalMeshValueCollection() = default;

  /// Return dimension of cell entity
  std::size_t dim() const { return _dim; }

  /// Return data
  const std::vector<std::pair<std::pair<std::size_t, std::size_t>, T>>&
  values() const
  {
    return _values;
  }

private:
  // Topological dimension
  const std::size_t _dim;

  // MeshValueCollection values (cell_index, local_index), value))
  std::vector<std::pair<std::pair<std::size_t, std::size_t>, T>> _values;

  // MPI communicator
  dolfin::MPI::Comm _mpi_comm;
};

//---------------------------------------------------------------------------
// Implementation of LocalMeshValueCollection
//---------------------------------------------------------------------------
template <typename T>
LocalMeshValueCollection<T>::LocalMeshValueCollection(
    MPI_Comm comm, const MeshValueCollection<T>& values, std::size_t dim)
    : _dim(dim), _mpi_comm(comm)
{
  // Prepare data
  std::vector<std::vector<std::size_t>> send_indices;
  std::vector<std::vector<T>> send_v;

  // Extract data on process 0 and split among processes
  if (MPI::rank(_mpi_comm.comm()) == 0)
  {
    // Get number of processes
    const std::size_t num_processes = MPI::size(_mpi_comm.comm());
    send_indices.resize(num_processes);
    send_v.resize(num_processes);

    const std::map<std::pair<std::size_t, std::size_t>, T>& vals
        = values.values();
    for (std::size_t p = 0; p < num_processes; p++)
    {
      const std::pair<std::size_t, std::size_t> local_range
          = MPI::local_range(_mpi_comm.comm(), p, vals.size());
      typename std::map<std::pair<std::size_t, std::size_t>, T>::const_iterator
          it
          = vals.begin();
      std::advance(it, local_range.first);
      for (std::size_t i = local_range.first; i < local_range.second; ++i)
      {
        send_indices[p].push_back(it->first.first);
        send_indices[p].push_back(it->first.second);
        send_v[p].push_back(it->second);
        std::advance(it, 1);
      }
    }
  }

  // Scatter data
  std::vector<std::size_t> indices;
  std::vector<T> v;
  MPI::scatter(_mpi_comm.comm(), send_indices, indices);
  MPI::scatter(_mpi_comm.comm(), send_v, v);
  assert(2 * v.size() == indices.size());

  // Unpack
  for (std::size_t i = 0; i < v.size(); ++i)
  {
    const std::size_t cell_index = indices[2 * i];
    const std::size_t local_entity_index = indices[2 * i + 1];
    const T value = v[i];
    _values.push_back({{cell_index, local_entity_index}, value});
  }
}
//---------------------------------------------------------------------------
}
}