// Copyright (C) 2009-2011 Johan Hake
//
// This file is part of DOLFIN.
//
// DOLFIN is free software: you can redistribute it and/or modify
// it under the terms of the GNU Lesser General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// DOLFIN is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with DOLFIN. If not, see <http://www.gnu.org/licenses/>.
//
// Modified by Anders Logg 2011, 2014
// Modified by Garth N. Wells 2013
//
// First added:  2011-10-15
// Last changed: 2014-08-11

#ifdef HAS_PETSC
#include <petscversion.h>
#endif

#include "types.h"
#include "defines.h"

//-------------------------------------------------------------------------
std::string dolfin::dolfin_version()
{
  return std::string(DOLFIN_VERSION);
}
//-------------------------------------------------------------------------
std::string dolfin::ufc_signature()
{
  return std::string(UFC_SIGNATURE);
}
//-------------------------------------------------------------------------
std::string dolfin::git_commit_hash()
{
  return std::string(DOLFIN_GIT_COMMIT_HASH);
}
//-------------------------------------------------------------------------
std::size_t dolfin::sizeof_la_index()
{
  return sizeof(dolfin::la_index);
}
//-------------------------------------------------------------------------
bool dolfin::has_openmp()
{
#ifdef HAS_OPENMP
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
bool dolfin::has_umfpack()
{
#ifdef HAS_UMFPACK
  return true;
#else
  return false;
#endif
}
//-------------------------------------------------------------------------
bool dolfin::has_cholmod()
{
#ifdef HAS_CHOLMOD
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
//-------------------------------------------------------------------------
bool dolfin::has_vtk()
{
#ifdef HAS_VTK
  return true;
#else
  return false;
#endif
}
//-------------------------------------------------------------------------
