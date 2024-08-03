// Copyright (C) 2024 Abdullah Mujahid
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

namespace dolfinx::io::checkpointing
{

/// @brief Write mesh to a file.
///
/// @param[in] ADIOS2 ADIOS2Wrapper
/// @param[in] mesh Mesh of type float or double to write to the file
template <std::floating_point T>
void write_mesh(ADIOS2Wrapper& ADIOS2, dolfinx::mesh::Mesh<T>& mesh);

/// @brief Read mesh from a file.
///
/// @param[in] ADIOS2 ADIOS2Wrapper
/// @param[in] comm comm
/// @return mesh reconstructed from the data
template <std::floating_point T>
dolfinx::mesh::Mesh<T> read_mesh(ADIOS2Wrapper& ADIOS2,
                                 MPI_Comm comm = MPI_COMM_WORLD);

/// @brief Read mesh from a file.
///
/// @param[in] _ADIOS2 ADIOS2Wrapper
/// @param[in] ADIOS2 ADIOS2Wrapper
/// @param[in] comm comm
/// @return mesh reconstructed from the data
std::variant<dolfinx::mesh::Mesh<float>, dolfinx::mesh::Mesh<double>>
read_mesh_variant(ADIOS2Wrapper& _ADIOS2, ADIOS2Wrapper& ADIOS2,
                  MPI_Comm comm = MPI_COMM_WORLD);

/// @brief Query floating_point type stored in the file
///
/// @param[in] ADIOS2 ADIOS2Wrapper
/// @return type float or double as a string
std::string query_type(ADIOS2Wrapper& ADIOS2);

} // namespace dolfinx::io::checkpointing

#endif
