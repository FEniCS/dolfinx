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
#include <mpi.h>

/// @file checkpointing.h
/// @brief ADIOS2 based checkpointing

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
/// @return mesh reconstructed from the data
template <std::floating_point T>
dolfinx::mesh::Mesh<T> read_mesh(adios2::IO& io, adios2::Engine& engine,
                                 MPI_Comm comm = MPI_COMM_WORLD);

} // namespace dolfinx::io::native

namespace dolfinx::io::impl_native
{
/// @brief Read mesh from a file.
///
/// @param[in] io ADIOS2 IO
/// @param[in] engine ADIOS2 Engine
/// @param[in] comm comm
/// @return mesh reconstructed from the data
std::variant<dolfinx::mesh::Mesh<float>, dolfinx::mesh::Mesh<double>>
read_mesh_variant(adios2::IO& io, adios2::Engine& engine,
                  MPI_Comm comm = MPI_COMM_WORLD);

} // namespace dolfinx::io::impl_native

#endif
