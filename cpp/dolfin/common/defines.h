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

/// Return sizeof the dolfin::la_index_t type
std::size_t sizeof_la_index_t();

/// Return true if DOLFIN is compiled in debugging mode,
/// i.e., with assertions on
bool has_debug();

/// Return true if DOLFIN is compiled with MPI
bool has_mpi();

/// Return true if DOLFIN is compiled with PETSc
bool has_petsc();

/// Return true if DOLFIN is compiled with SLEPc
bool has_slepc();

/// Return true if DOLFIN is compiled with Scotch
bool has_scotch();

/// Return true if DOLFIN is compiled with ParMETIS
bool has_parmetis();

/// Return true if DOLFIN is compiled with ZLIB
bool has_zlib();

/// Return true if DOLFIN is compiled with HDF5
bool has_hdf5();

/// Return true if DOLFIN is compiled with Parallel HDF5
bool has_hdf5_parallel();
}


