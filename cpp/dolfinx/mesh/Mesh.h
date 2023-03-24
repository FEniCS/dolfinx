// Copyright (C) 2006-2020 Anders Logg, Chris Richardson and Garth N. Wells
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include "Geometry.h"
#include "Mesh.h"
#include "Topology.h"
#include "cell_types.h"
#include <algorithm>
#include <concepts>
#include <dolfinx/common/IndexMap.h>
#include <dolfinx/common/MPI.h>
#include <dolfinx/fem/CoordinateElement.h>
#include <dolfinx/graph/AdjacencyList.h>
#include <dolfinx/graph/ordering.h>
#include <dolfinx/graph/partition.h>
#include <memory>
#include <string>
#include <utility>

namespace dolfinx::mesh
{

// See https://github.com/doxygen/doxygen/issues/9552
/// Signature for the cell partitioning function. The function should
/// compute the destination rank for cells currently on this rank.
using CellPartitionFunction = std::function<graph::AdjacencyList<std::int32_t>(
    MPI_Comm comm, int nparts, int tdim,
    const graph::AdjacencyList<std::int64_t>& cells)>;

/// A Mesh consists of a set of connected and numbered mesh topological
/// entities, and geometry data
template <typename T>
class Mesh
{
public:
  /// @brief Value type
  using value_type = T;

  /// Create a mesh
  /// @param[in] comm MPI Communicator
  /// @param[in] topology Mesh topology
  /// @param[in] geometry Mesh geometry
  template <std::convertible_to<Topology> U, std::convertible_to<Geometry<T>> V>
  Mesh(MPI_Comm comm, U&& topology, V&& geometry)
      : _topology(std::forward<U>(topology)),
        _geometry(std::forward<V>(geometry)), _comm(comm)
  {
    // Do nothing
  }

  /// Copy constructor
  /// @param[in] mesh Mesh to be copied
  Mesh(const Mesh& mesh) = default;

  /// Move constructor
  /// @param mesh Mesh to be moved.
  Mesh(Mesh&& mesh) = default;

  /// Destructor
  ~Mesh() = default;

  // Assignment operator
  Mesh& operator=(const Mesh& mesh) = delete;

  /// Assignment move operator
  /// @param mesh Another Mesh object
  Mesh& operator=(Mesh&& mesh) = default;

  // TODO: Is there any use for this? In many situations one has to get the
  // topology of a const Mesh, which is done by Mesh::topology_mutable. Note
  // that the python interface (calls Mesh::topology()) may still rely on it.
  /// Get mesh topology
  /// @return The topology object associated with the mesh.
  Topology& topology() { return _topology; }

  /// Get mesh topology (const version)
  /// @return The topology object associated with the mesh.
  const Topology& topology() const { return _topology; }

  /// Get mesh topology if one really needs the mutable version
  /// @return The topology object associated with the mesh.
  Topology& topology_mutable() const { return _topology; }

  /// Get mesh geometry
  /// @return The geometry object associated with the mesh
  Geometry<T>& geometry() { return _geometry; }

  /// Get mesh geometry (const version)
  /// @return The geometry object associated with the mesh
  const Geometry<T>& geometry() const { return _geometry; }

  /// Mesh MPI communicator
  /// @return The communicator on which the mesh is distributed
  MPI_Comm comm() const { return _comm.comm(); }

  /// Name
  std::string name = "mesh";

private:
  // Mesh topology:
  // TODO: This is mutable because of the current memory management within
  // mesh::Topology. It allows to obtain a non-const Topology from a
  // const mesh (via Mesh::topology_mutable()).
  //
  mutable Topology _topology;

  // Mesh geometry
  Geometry<T> _geometry;

  // MPI communicator
  dolfinx::MPI::Comm _comm;
};

} // namespace dolfinx::mesh
