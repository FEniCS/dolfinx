// Copyright (C) year authors
//
// This file is part of DOLFINX (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#ifdef HAS_ADIOS2

#include <adios2.h>
#include <basix/finite-element.h>
#include <dolfinx/mesh/Mesh.h>
#include <mpi.h>

/// @file checkpointing.h
/// @brief ADIOS2 based checkpointing

namespace dolfinx::io::checkpointing
{

void test() { std::cout << "HELLO" << std::endl; };

void write(MPI_Comm comm, std::string filename, std::string tag,
           std::shared_ptr<dolfinx::mesh::Mesh<float>> mesh);

} // namespace dolfinx::io::checkpointing

#endif
