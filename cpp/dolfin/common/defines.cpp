// Copyright (C) 2009-2011 Johan Hake
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "defines.h"
#include "types.h"
#include <hdf5.h>
#include <petscversion.h>

//-------------------------------------------------------------------------
std::string dolfin::dolfin_version() { return std::string(DOLFIN_VERSION); }
//-------------------------------------------------------------------------
std::string dolfin::ufc_signature() { return std::string(UFC_SIGNATURE); }
//-------------------------------------------------------------------------
std::string dolfin::git_commit_hash()
{
  return std::string(DOLFIN_GIT_COMMIT_HASH);
}
//-------------------------------------------------------------------------
bool dolfin::has_debug()
{
#ifdef DEBUG
  return true;
#else
  return false;
#endif
}
//-------------------------------------------------------------------------
bool dolfin::has_petsc_complex()
{
#ifdef PETSC_USE_COMPLEX
  return true;
#else
  return false;
#endif
}
//-------------------------------------------------------------------------
bool dolfin::has_slepc()
{
#ifdef HAS_SLEPC
  return true;
#else
  return false;
#endif
}
//-------------------------------------------------------------------------
bool dolfin::has_parmetis()
{
#ifdef HAS_PARMETIS
  return true;
#else
  return false;
#endif
}
//-------------------------------------------------------------------------
bool dolfin::has_kahip()
{
#ifdef HAS_KAHIP
  return true;
#else
  return false;
#endif
}
