// Copyright (C) year authors
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

void write(MPI_Comm comm, std::string filename, std::string tag,
           //    adios2::IO io, adios2::Engine engine,
           std::shared_ptr<dolfinx::mesh::Mesh<float>> mesh);

void write(MPI_Comm comm, std::string filename, std::string tag,
           //    adios2::IO io, adios2::Engine engine,
           std::shared_ptr<dolfinx::mesh::Mesh<double>> mesh);

} // namespace dolfinx::io::checkpointing

#endif
