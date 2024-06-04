// Copyright (C) 2009-2011 Johan Hake
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include <string>

#include "version.h"

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
consteval bool has_debug()
{
#ifndef NDEBUG
  return true;
#else
  return false;
#endif
}

/// Return true if DOLFINx is compiled with PETSc
consteval bool has_petsc()
{
#ifdef HAS_PETSC
  return true;
#else
  return false;
#endif
}

/// Return true if DOLFINx is compiled with SLEPc
consteval bool has_slepc()
{
#ifdef HAS_SLEPC
  return true;
#else
  return false;
#endif
}

/// Return true if DOLFINx is compiled with ParMETIS
consteval bool has_parmetis()
{
#ifdef HAS_PARMETIS
  return true;
#else
  return false;
#endif
}

/// Return true if DOLFINx is compiled with KaHIP
consteval bool has_kahip()
{
#ifdef HAS_KAHIP
  return true;
#else
  return false;
#endif
}

/// Return true if DOLFINX is compiled with ADIOS2
consteval bool has_adios2()
{
#ifdef HAS_ADIOS2
  return true;
#else
  return false;
#endif
}

} // namespace dolfinx
