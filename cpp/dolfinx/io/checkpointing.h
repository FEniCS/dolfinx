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
#include <dolfinx/mesh/MeshTags.h>
#include <dolfinx/mesh/utils.h>
#include <mpi.h>

/// @file checkpointing.h
/// @brief ADIOS2 based checkpointing

namespace dolfinx::io::checkpointing
{

/// @brief Write mesh to a file.
///
/// @param[in] adios2engine ADIOS2Engine
/// @param[in] mesh Mesh of type float to write to the file
void write_mesh(ADIOS2Engine& adios2engine,
                std::shared_ptr<dolfinx::mesh::Mesh<float>> mesh);

/// @brief Write mesh to a file.
///
/// @param[in] adios2engine ADIOS2Engine
/// @param[in] mesh Mesh of type double to write to the file
void write_mesh(ADIOS2Engine& adios2engine,
                std::shared_ptr<dolfinx::mesh::Mesh<double>> mesh);

/// @brief Write meshtags to a file.
///
/// @param[in] adios2engine ADIOS2Engine
/// @param[in] mesh Mesh of type float to write to the file
/// @param[in] meshtags MeshTags of type float to write to the file
void write_meshtags(ADIOS2Engine& adios2engine,
                    std::shared_ptr<dolfinx::mesh::Mesh<float>> mesh,
                    std::shared_ptr<dolfinx::mesh::MeshTags<float>> meshtags);

/// @brief Write meshtags to a file.
///
/// @param[in] adios2engine ADIOS2Engine
/// @param[in] mesh Mesh of type double to write to the file
/// @param[in] meshtags MeshTags of type double to write to the file
void write_meshtags(ADIOS2Engine& adios2engine,
                    std::shared_ptr<dolfinx::mesh::Mesh<double>> mesh,
                    std::shared_ptr<dolfinx::mesh::MeshTags<double>> meshtags);

} // namespace dolfinx::io::checkpointing

#endif
