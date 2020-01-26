// Copyright (C) 2009-2011 Johan Hake
//
// This file is part of DOLFINX (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "defines.h"
#include "types.h"
#include <hdf5.h>
#include <petscversion.h>

//-------------------------------------------------------------------------
std::string dolfinx::version() { return std::string(DOLFINX_VERSION); }
//-------------------------------------------------------------------------
std::string dolfinx::ufc_signature() { return std::string(UFC_SIGNATURE); }
//-------------------------------------------------------------------------
std::string dolfinx::git_commit_hash()
{
  return std::string(DOLFINX_GIT_COMMIT_HASH);
}
//-------------------------------------------------------------------------
bool dolfinx::has_debug()
{
#ifdef DEBUG
  return true;
#else
  return false;
#endif
}
//-------------------------------------------------------------------------
bool dolfinx::has_petsc_complex()
{
#ifdef PETSC_USE_COMPLEX
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
