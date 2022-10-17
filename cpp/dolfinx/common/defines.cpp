// Copyright (C) 2009-2011 Johan Hake
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "defines.h"

//-------------------------------------------------------------------------
std::string dolfinx::version() { return std::string(DOLFINX_VERSION); }
//-------------------------------------------------------------------------
std::string dolfinx::ufcx_signature() { return std::string(UFCX_SIGNATURE); }
//-------------------------------------------------------------------------
std::string dolfinx::git_commit_hash()
{
  return std::string(DOLFINX_GIT_COMMIT_HASH);
}
//-------------------------------------------------------------------------
bool dolfinx::has_debug()
{
#ifndef NDEBUG
  return true;
#else
  return false;
#endif
}
//-------------------------------------------------------------------------
bool dolfinx::has_slepc()
{
#ifdef HAS_SLEPC
  return true;
#else
  return false;
#endif
}
//-------------------------------------------------------------------------
bool dolfinx::has_parmetis()
{
#ifdef HAS_PARMETIS
  return true;
#else
  return false;
#endif
}
//-------------------------------------------------------------------------
bool dolfinx::has_kahip()
{
#ifdef HAS_KAHIP
  return true;
#else
  return false;
#endif
}
//-------------------------------------------------------------------------
bool dolfinx::has_adios2()
{
#ifdef HAS_ADIOS2
  return true;
#else
  return false;
#endif
}
//-------------------------------------------------------------------------
