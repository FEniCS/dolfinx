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
// Modified by Anders Logg, 2011
// Modified by Garth N. Wells, 2013
//
// First added:  2009-09-03
// Last changed: 2014-08-11

#ifndef __DOLFIN_DEFINES_H
#define __DOLFIN_DEFINES_H

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

  /// Return sizeof the dolfin::la_index type
  std::size_t sizeof_la_index();

  /// Return true if DOLFIN is compiled with OpenMP
  bool has_openmp();

  /// Return true if DOLFIN is compiled with MPI
  bool has_mpi();

  /// Return true if DOLFIN is compiled with PETSc
  bool has_petsc();

  /// Return true if DOLFIN is compiled with SLEPc
  bool has_slepc();

  /// Return true if DOLFIN is compiled with Scotch
  bool has_scotch();

  /// Return true if DOLFIN is compiled with Umfpack
  bool has_umfpack();

  /// Return true if DOLFIN is compiled with Cholmod
  bool has_cholmod();

  /// Return true if DOLFIN is compiled with ParMETIS
  bool has_parmetis();

  /// Return true if DOLFIN is compiled with ZLIB
  bool has_zlib();

  /// Return true if DOLFIN is compiled with HDF5
  bool has_hdf5();

}

#endif
