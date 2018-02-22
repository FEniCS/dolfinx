// Copyright (C) 2009-2011 Johan Hake
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#ifdef HAS_PETSC
#include <petscversion.h>
#endif

#ifdef HAS_HDF5
#include <hdf5.h>
#endif

#include "defines.h"
#include "types.h"

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
std::size_t dolfin::sizeof_la_index_t() { return sizeof(dolfin::la_index_t); }
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
bool dolfin::has_mpi()
{
#ifdef HAS_MPI
  return true;
#else
  return false;
#endif
}
//-------------------------------------------------------------------------
bool dolfin::has_petsc()
{
#ifdef HAS_PETSC
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
bool dolfin::has_scotch()
{
#ifdef HAS_SCOTCH
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
bool dolfin::has_zlib()
{
#ifdef HAS_ZLIB
  return true;
#else
  return false;
#endif
}
//-------------------------------------------------------------------------
bool dolfin::has_hdf5()
{
#ifdef HAS_HDF5
  return true;
#else
  return false;
#endif
}
//-----------------------------------------------------------------------------
bool dolfin::has_hdf5_parallel()
{
#ifdef H5_HAVE_PARALLEL
  return true;
#else
  return false;
#endif
}
//-------------------------------------------------------------------------
