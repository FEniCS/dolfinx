// Copyright (C) 2026 Jack S. Hale
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include <catch2/catch_test_macros.hpp>
#include <dolfinx/common/MPI.h>
#include <dolfinx/io/XDMFFile.h>
#include <filesystem>
#include <fstream>
#include <hdf5.h>
#include <mpi.h>
#include <utility>

using namespace dolfinx;

namespace
{
/// Number of HDF5 files that are currently open
ssize_t open_files() { return H5Fget_obj_count(H5F_OBJ_ALL, H5F_OBJ_FILE); }
} // namespace

TEST_CASE("XDMFFile move constructor releases the HDF5 handle", "[xdmf]")
{
  const ssize_t before = open_files();
  {
    io::XDMFFile file(MPI_COMM_WORLD, "xdmf_move.xdmf", "w");
    io::XDMFFile moved(std::move(file));
    CHECK(open_files() == before + 1);
    moved.close();
  }

  // The moved-from object must not close the file a second time
  CHECK(open_files() == before);
}

TEST_CASE("XDMFFile move assignment releases the HDF5 handle", "[xdmf]")
{
  const ssize_t before = open_files();
  {
    io::XDMFFile file(MPI_COMM_WORLD, "xdmf_move_assign_a.xdmf", "w");
    io::XDMFFile other(MPI_COMM_WORLD, "xdmf_move_assign_b.xdmf", "w");
    CHECK(open_files() == before + 2);

    // Assigning over `other` must close the file it held
    other = std::move(file);
    CHECK(open_files() == before + 1);
  }
  CHECK(open_files() == before);
}

TEST_CASE("XDMFFile constructor does not leak on failure", "[xdmf]")
{
  // Create the HDF5 file that opening in "r" mode requires
  {
    io::XDMFFile file(MPI_COMM_WORLD, "xdmf_ctor_throw.xdmf", "w");
    file.close();
  }

  // Make the XDMF file unparseable, so that the constructor throws
  // after it has opened the HDF5 file
  if (dolfinx::MPI::rank(MPI_COMM_WORLD) == 0)
    std::ofstream("xdmf_ctor_throw.xdmf") << "not xml";
  MPI_Barrier(MPI_COMM_WORLD);

  const ssize_t before = open_files();
  CHECK_THROWS(io::XDMFFile(MPI_COMM_WORLD, "xdmf_ctor_throw.xdmf", "r"));

  // A constructor that throws has no destructor call, so the HDF5 file
  // is released only if the handle owns it
  CHECK(open_files() == before);
}
