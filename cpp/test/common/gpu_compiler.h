// Copyright (C) 2026 Garth N. Wells
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include <mpi.h>

namespace dolfinx_test
{
/// Outcomes of the checks in gpu_compiler_host_code.
struct GpuCompilerResults
{
  bool interpolation;    ///< Interpolated values are correct
  bool submesh;          ///< EntityMap round trip recovers the sub-cells
  bool assembly;         ///< Non-zero stiffness matrix annihilates constants
  bool assembly_with_bc; ///< Constrained rows have unit diagonal only
};

/// @brief Run DOLFINx host code compiled by a GPU compiler (nvcc or
/// HIP).
/// @param[in] comm MPI communicator.
/// @return Outcome of each check.
GpuCompilerResults gpu_compiler_host_code(MPI_Comm comm);
} // namespace dolfinx_test
