// Copyright (C) year authors
//
// This file is part of DOLFINX (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#ifdef HAS_ADIOS2

#include <mpi.h>
#include <adios2.h>
#include <dolfinx/mesh/Mesh.h>
#include <basix/finite-element.h>

/// @file checkpointing.h
/// @brief ADIOS2 based checkpointing

namespace dolfinx::io::checkpointing
{

template <std::floating_point T>
void write(MPI_Comm comm, std::string filename,
           std::string tag, std::shared_ptr<mesh::Mesh<T>> mesh);


} //namespace dolfinx::io::checkpointing

#endif