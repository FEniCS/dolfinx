// Copyright (C) 2009-2011 Johan Hake
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include <string>

namespace dolfinx
{

/// Return DOLFINx version string
std::string version();

/// Return UFC signature string
std::string ufcx_signature();

/// Return git changeset hash (returns "unknown" if changeset is
/// not known)
std::string git_commit_hash();

/// Return true if DOLFINx is compiled in debugging mode,
/// i.e., with assertions on
bool has_debug();

/// Return true if DOLFINx is compiled with PETSc
bool has_petsc();

/// Return true if DOLFINx is compiled with SLEPc
bool has_slepc();

/// Return true if DOLFINx is compiled with Scotch
bool has_scotch();

/// Return true if DOLFINx is compiled with ParMETIS
bool has_parmetis();

/// Return true if DOLFINx is compiled with KaHIP
bool has_kahip();

/// Return true if DOLFINX is compiled with ADIOS2
bool has_adios2();

} // namespace dolfinx
