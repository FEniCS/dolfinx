// Copyright (C) 2026 Jack S. Hale
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include <catch2/catch_test_macros.hpp>
#include <dolfinx/io/XDMFFile.h>
#include <filesystem>
#include <mpi.h>
#include <utility>

using namespace dolfinx;

TEST_CASE("XDMFFile move constructor releases the HDF5 handle", "[xdmf]")
{
  {
    io::XDMFFile file(MPI_COMM_WORLD, "xdmf_move.xdmf", "w");
    io::XDMFFile moved(std::move(file));
    moved.close();
  }

  // Reaching this point shows the moved-from object did not close the
  // file a second time. io::hdf5::close_file throws on failure, so a
  // second close would escape ~XDMFFile and terminate the process.
  SUCCEED("Moved-from XDMFFile did not close the HDF5 file");
}
