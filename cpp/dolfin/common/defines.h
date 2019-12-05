// Copyright (C) 2009-2011 Johan Hake
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include <string>

namespace dolfin
{

/// Return DOLFIN version string
std::string dolfin_version();

/// Return UFC signature string
std::string ufc_signature();

/// Return git changeset hash (returns "unknown" if changeset is
/// not known)
std::string git_commit_hash();

/// Return true if DOLFIN is compiled in debugging mode,
/// i.e., with assertions on
bool has_debug();

/// Return true if DOLFIN is configured with PETSc compiled
/// with scalars represented as complex numbers
bool has_petsc_complex();

/// Return true if DOLFIN is compiled with SLEPc
bool has_slepc();

/// Return true if DOLFIN is compiled with Scotch
bool has_scotch();

/// Return true if DOLFIN is compiled with ParMETIS
bool has_parmetis();

/// Return true if DOLFIN is compiled with KaHIP
bool has_kahip();

} // namespace dolfin
