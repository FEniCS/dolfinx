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

/// Return true if DOLFINX is compiled with PT-SCOTCH
consteval bool has_ptscotch()
{
#ifdef HAS_PTSCOTCH
  return true;
#else
  return false;
#endif
}

/// Return true if DOLFINx supports UFCx kernels with arguments of type C99
/// _Complex. When DOLFINx was built with MSVC this returns false. This
/// returning false does not preclude using DOLFINx with kernels accepting
/// std::complex.
consteval bool has_complex_ufcx_kernels()
{
#ifdef DOLFINX_NO_STDC_COMPLEX_KERNELS
  return false;
#else
  return true;
#endif
}

} // namespace dolfinx
