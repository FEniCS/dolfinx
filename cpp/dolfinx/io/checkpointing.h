// Copyright (C) 2024 Abdullah Mujahid
//
// This file is part of DOLFINX (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#ifdef HAS_ADIOS2

#include "boost/variant.hpp"
#include <adios2.h>
#include <basix/finite-element.h>
#include <dolfinx/mesh/Mesh.h>
#include <mpi.h>

/// @file checkpointing.h
/// @brief ADIOS2 based checkpointing

namespace dolfinx::io::checkpointing
{

typedef boost::variant<float, double> floating_point;

/// @brief Write mesh to a file.
///
/// @param[in] io ADIOS2 IO
/// @param[in] engine ADIOS2 Engine
/// @param[in] mesh Mesh of type float or double to write to the file
template <std::floating_point T>
void write_mesh(adios2::IO& io, adios2::Engine& engine,
                dolfinx::mesh::Mesh<T>& mesh);

/// @brief Write mesh to a file.
///
/// @param[in] io ADIOS2 IO
/// @param[in] engine ADIOS2 Engine
/// @param[in] comm comm
/// @return mesh reconstructed from the data
dolfinx::mesh::Mesh<float> read_mesh(adios2::IO& io, adios2::Engine& engine,
                                     MPI_Comm comm = MPI_COMM_WORLD);

} // namespace dolfinx::io::checkpointing

#endif
