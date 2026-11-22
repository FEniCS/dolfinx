// Copyright (C) 2026 Garth N. Wells
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include <mpi.h>

namespace dolfinx_test
{
/// Outcomes of the checks in cuda_nvcc_host_code.
struct CudaNvccResults
{
  bool interpolation;    ///< Interpolated values are correct
  bool submesh;          ///< EntityMap round trip recovers the sub-cells
  bool assembly;         ///< Stiffness matrix annihilates constants
  bool assembly_with_bc; ///< Assembly with a Dirichlet BC is non-zero
};

/// @brief Run DOLFINx host code compiled by nvcc.
/// @param[in] comm MPI communicator.
/// @return Outcome of each check.
CudaNvccResults cuda_nvcc_host_code(MPI_Comm comm);
} // namespace dolfinx_test
