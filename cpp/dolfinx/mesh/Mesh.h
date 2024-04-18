// Copyright (C) 2006-2023 Anders Logg, Chris Richardson and Garth N. Wells
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include "Geometry.h"
#include <concepts>
#include <dolfinx/common/MPI.h>
#include <string>

namespace dolfinx::mesh
{
class Topology;

/// @brief A Mesh consists of a set of connected and numbered mesh
/// topological entities, and geometry data.
/// @tparam The floating point type for representing the geometry.
template <std::floating_point T>
class Mesh
{
public:
  /// @brief Value type
  using geometry_type = Geometry<T>;

  /// @brief Create a mesh
  /// @param[in] comm MPI Communicator
  /// @param[in] topology Mesh topology
  /// @param[in] geometry Mesh geometry
  template <typename V>
    requires std::is_convertible_v<std::remove_cvref_t<V>, Geometry<T>>
  Mesh(MPI_Comm comm, std::shared_ptr<Topology> topology, V&& geometry)
      : _topology(topology), _geometry(std::forward<V>(geometry)), _comm(comm)
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

  // TODO: Is there any use for this? In many situations one has to get
  // the topology of a const Mesh, which is done by
  // Mesh::topology_mutable. Note that the python interface (calls
  // Mesh::topology()) may still rely on it.
  /// @brief Get mesh topology
  /// @return The topology object associated with the mesh.
  std::shared_ptr<Topology> topology() { return _topology; }

  /// Get mesh topology (const version)
  /// @return The topology object associated with the mesh.
  std::shared_ptr<const Topology> topology() const { return _topology; }

  /// Get mesh topology if one really needs the mutable version
  /// @return The topology object associated with the mesh.
  std::shared_ptr<Topology> topology_mutable() const { return _topology; }

  /// @brief Get mesh geometry
  /// @return The geometry object associated with the mesh
  Geometry<T>& geometry() { return _geometry; }

  /// @brief Get mesh geometry (const version)
  /// @return The geometry object associated with the mesh
  const Geometry<T>& geometry() const { return _geometry; }

  /// @brief Mesh MPI communicator
  /// @return The communicator on which the mesh is distributed
  MPI_Comm comm() const { return _comm.comm(); }

  /// Name
  std::string name = "mesh";

private:
  // Mesh topology
  // Note: This is mutable because of the current memory management
  // within mesh::Topology. It allows to obtain a non-const Topology
  // from a const mesh (via Mesh::topology_mutable()).
  std::shared_ptr<Topology> _topology;

  // Mesh geometry
  Geometry<T> _geometry;

  // MPI communicator
  dolfinx::MPI::Comm _comm;
};

/// @cond
/// Template type deduction
template <typename V>
Mesh(MPI_Comm, std::shared_ptr<Topology>,
     V) -> Mesh<typename std::remove_cvref_t<typename V::value_type>>;
/// @endcond

} // namespace dolfinx::mesh
