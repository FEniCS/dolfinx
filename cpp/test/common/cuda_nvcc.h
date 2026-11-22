// Copyright (C) 2026 Garth N. Wells
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include <mpi.h>

namespace dolfinx_test
{
/// @brief Run DOLFINx host code compiled by nvcc.
/// @param[in] comm MPI communicator.
/// @return True if the results are as expected.
bool cuda_nvcc_host_code(MPI_Comm comm);
} // namespace dolfinx_test
