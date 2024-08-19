// Copyright (C) 2024 Abdullah Mujahid, Jørgen S. Dokken, Jack S. Hale
//
// This file is part of DOLFINX (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#ifdef HAS_ADIOS2

#include "ADIOS2_utils.h"
#include <adios2.h>
#include <basix/finite-element.h>
#include <dolfinx/mesh/Mesh.h>
#include <dolfinx/mesh/utils.h>
#include <mpi.h>

/// @file checkpointing.h
/// @brief ADIOS2 based checkpointing

namespace dolfinx::io::impl_native
{
/// @brief Find offset and size.
///
/// @param[in] rank MPI rank
/// @param[in] N size of data to distribute
/// @param[in] size MPI size
/// @return start and count
std::pair<std::uint64_t, std::uint64_t> get_counters(int rank, std::uint64_t N,
                                                     int size);

/// @brief Read geometry data
///
/// @param[in] io ADIOS2 IO
/// @param[in] engine ADIOS2 Engine
/// @param[in] dim The geometric dimension (`0 < dim <= 3`).
/// @param[in] num_nodes_global size of the global array of nodes
/// @param[in] rank MPI rank
/// @param[in] size MPI size
/// @return The point coordinates of row-major storage and
/// itsshape `(num_nodes_local, dim)`
template <std::floating_point T>
std::pair<std::vector<T>, std::array<std::size_t, 2>>
read_geometry_data(adios2::IO& io, adios2::Engine& engine, int dim,
                   std::uint64_t num_nodes_global, int rank, int size);

/// @brief Read topology array
///
/// @param[in] io ADIOS2 IO
/// @param[in] engine ADIOS2 Engine
/// @param[in] num_cells_global global number of cells
/// @param[in] rank MPI rank
/// @param[in] size MPI size
/// @return The cell-to-node connectivity in a flattened array
std::vector<int64_t> read_topology_data(adios2::IO& io, adios2::Engine& engine,
                                        std::uint64_t num_cells_global,
                                        int rank, int size);

/// @brief Read mesh from a file.
///
/// @param[in] io ADIOS2 IO
/// @param[in] engine ADIOS2 Engine
/// @param[in] comm comm
/// @param[in] ghost_mode The requested type of cell ghosting/overlap
/// @return mesh reconstructed from the data
std::variant<dolfinx::mesh::Mesh<float>, dolfinx::mesh::Mesh<double>>
read_mesh_variant(adios2::IO& io, adios2::Engine& engine,
                  MPI_Comm comm = MPI_COMM_WORLD,
                  dolfinx::mesh::GhostMode ghost_mode
                  = dolfinx::mesh::GhostMode::shared_facet);

} // namespace dolfinx::io::impl_native

namespace dolfinx::io::native
{

/// @brief Write mesh to a file.
///
/// @param[in] io ADIOS2 IO
/// @param[in] engine ADIOS2 Engine
/// @param[in] mesh Mesh of type float or double to write to the file
template <std::floating_point T>
void write_mesh(adios2::IO& io, adios2::Engine& engine,
                const dolfinx::mesh::Mesh<T>& mesh);

/// @brief Read mesh from a file.
///
/// @param[in] io ADIOS2 IO
/// @param[in] engine ADIOS2 Engine
/// @param[in] comm comm
/// @param[in] ghost_mode The requested type of cell ghosting/overlap
/// @return mesh reconstructed from the data
template <std::floating_point T>
dolfinx::mesh::Mesh<T> read_mesh(adios2::IO& io, adios2::Engine& engine,
                                 MPI_Comm comm = MPI_COMM_WORLD,
                                 dolfinx::mesh::GhostMode ghost_mode
                                 = dolfinx::mesh::GhostMode::shared_facet);

} // namespace dolfinx::io::native

#endif
