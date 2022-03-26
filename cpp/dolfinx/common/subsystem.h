// Copyright (C) 2008-2020 Garth N. Wells
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

namespace dolfinx::common
{

/// Functions in this namesspace are convenience functtions for the
/// initialisation and finalisation of various sub systems, such as MPI
/// and PETSc.
namespace subsystem
{

/// Initialise loguru
void init_logging(int argc, char* argv[]);

} // namespace subsystem
} // namespace dolfinx::common
