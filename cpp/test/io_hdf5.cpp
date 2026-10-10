// Copyright (C) 2026 Garth N. Wells
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include <array>
#include <catch2/catch_test_macros.hpp>
#include <cstdint>
#include <dolfinx/common/MPI.h>
#include <dolfinx/io/HDF5Interface.h>
#include <filesystem>
#include <format>
#include <hdf5.h>
#include <mpi.h>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

using namespace dolfinx;

namespace
{
/// Per-rank filename, so that each test writes a file of its own
std::filesystem::path test_file(std::string_view name)
{
  return std::format("hdf5_{}_{}.h5", name, dolfinx::MPI::rank(MPI_COMM_WORLD));
}

/// Number of HDF5 identifiers that are currently open
ssize_t open_ids() { return H5Fget_obj_count(H5F_OBJ_ALL, H5F_OBJ_ALL); }
} // namespace

TEST_CASE("HDF5 filename round-trip", "[hdf5]")
{
  const std::filesystem::path filename = test_file("filename");
  hid_t handle = io::hdf5::open_file(MPI_COMM_SELF, filename, "w", false);
  const std::string name = io::hdf5::get_filename(handle).string();
  io::hdf5::close_file(handle);

  // The filename must not carry the null terminator of the buffer it
  // was read into, which would make the path longer than its own C
  // string and truncate anything appended to it
  CHECK(name == filename.string());
  CHECK(name.size() == std::string(name.c_str()).size());
}

TEST_CASE("HDF5 attributes are replaced, not truncated", "[hdf5]")
{
  hid_t handle
      = io::hdf5::open_file(MPI_COMM_SELF, test_file("attr"), "w", false);

  SECTION("string attribute grows")
  {
    io::hdf5::set_attribute(handle, "Type", "abc");
    io::hdf5::set_attribute(handle, "Type", "UnstructuredGrid");

    io::hdf5::Handle attr(H5Aopen(handle, "Type", H5P_DEFAULT), H5Aclose);
    io::hdf5::Handle type(H5Aget_type(attr), H5Tclose);
    std::string value(H5Tget_size(type) + 1, '\0');
    REQUIRE(H5Aread(attr, type, value.data()) >= 0);
    CHECK(std::string(value.c_str()) == "UnstructuredGrid");
  }

  SECTION("array attribute shrinks")
  {
    // An attribute written over a longer one must not leave the
    // original dataspace in place, which would read past the end of the
    // new value
    io::hdf5::set_attribute(handle, "Version",
                            std::vector<std::int32_t>{1, 2, 3, 4, 5});
    io::hdf5::set_attribute(handle, "Version", std::vector<std::int32_t>{2, 2});

    io::hdf5::Handle attr(H5Aopen(handle, "Version", H5P_DEFAULT), H5Aclose);
    io::hdf5::Handle space(H5Aget_space(attr), H5Sclose);
    hsize_t dim = 0;
    H5Sget_simple_extent_dims(space, &dim, nullptr);
    CHECK(dim == 2);

    std::array<std::int32_t, 2> value{0, 0};
    REQUIRE(H5Aread(attr, H5T_NATIVE_INT32, value.data()) >= 0);
    CHECK(value == std::array<std::int32_t, 2>{2, 2});
  }

  io::hdf5::close_file(handle);
}

TEST_CASE("HDF5 invalid arguments are rejected", "[hdf5]")
{
  CHECK_THROWS_AS(
      io::hdf5::open_file(MPI_COMM_SELF, test_file("mode"), "rw", false),
      std::invalid_argument);

  hid_t handle
      = io::hdf5::open_file(MPI_COMM_SELF, test_file("missing"), "w", false);
  CHECK_THROWS_AS(io::hdf5::open_dataset(handle, "/does/not/exist"),
                  std::runtime_error);
  CHECK_THROWS_AS(io::hdf5::get_dataset_shape(handle, "/does/not/exist"),
                  std::runtime_error);

  // Only rank 1 and rank 2 datasets are supported
  std::array<double, 1> x{0};
  CHECK_THROWS_AS(io::hdf5::write_dataset(handle, "/x", x.data(), {0, 1},
                                          {1, 1, 1}, false, false),
                  std::invalid_argument);
  io::hdf5::close_file(handle);
}

TEST_CASE("HDF5 dataset round-trip", "[hdf5]")
{
  const std::vector<double> x{1, 2, 3, 4, 5, 6};
  hid_t handle
      = io::hdf5::open_file(MPI_COMM_SELF, test_file("data"), "w", false);
  io::hdf5::write_dataset(handle, "/group/x", x.data(), {0, 3}, {3, 2}, false,
                          false);

  CHECK(io::hdf5::has_dataset(handle, "/group/x"));
  CHECK(!io::hdf5::has_dataset(handle, "/group/y"));
  CHECK(io::hdf5::get_dataset_shape(handle, "/group/x")
        == std::vector<std::int64_t>{3, 2});

  // allow_cast = false requires the stored and requested types to match
  CHECK(io::hdf5::read_dataset<double>(handle, "/group/x", {0, 3}, false) == x);
  CHECK_THROWS_AS(
      io::hdf5::read_dataset<std::int64_t>(handle, "/group/x", {0, 3}, false),
      std::runtime_error);

  // Extend an existing (chunked) dataset
  io::hdf5::write_dataset(handle, "/group/y", x.data(), {0, 6}, {6}, false,
                          true);
  io::hdf5::write_dataset(handle, "/group/y", x.data(), {6, 12}, {12}, false,
                          true);
  CHECK(io::hdf5::get_dataset_shape(handle, "/group/y")
        == std::vector<std::int64_t>{12});

  io::hdf5::close_file(handle);
}

TEST_CASE("HDF5 identifiers are not leaked", "[hdf5]")
{
  const ssize_t before = open_ids();
  {
    const std::vector<double> x{1, 2, 3, 4};
    hid_t handle
        = io::hdf5::open_file(MPI_COMM_SELF, test_file("leak"), "w", false);
    io::hdf5::add_group(handle, "/a/b");
    io::hdf5::set_attribute(handle, "Name", "mesh");
    io::hdf5::set_attribute(handle, "Count", 4);
    io::hdf5::set_attribute(handle, "Shape", std::vector<std::int32_t>{2, 2});
    io::hdf5::write_dataset(handle, "/a/b/x", x.data(), {0, 2}, {2, 2}, false,
                            false);

    // The type check of a non-casting read must not retain the dataset
    // datatype
    CHECK(io::hdf5::read_dataset<double>(handle, "/a/b/x", {0, 2}, false) == x);
    CHECK_THROWS(
        io::hdf5::read_dataset<float>(handle, "/a/b/x", {0, 2}, false));
    io::hdf5::close_file(handle);
  }
  CHECK(open_ids() == before);
}

TEST_CASE("HDF5 scope guard closes its identifier", "[hdf5]")
{
  const ssize_t before = open_ids();
  {
    io::hdf5::Handle file(
        io::hdf5::open_file(MPI_COMM_SELF, test_file("close"), "w", false),
        H5Fclose);
    CHECK(open_ids() == before + 1);

    // close() empties the handle, so destruction does not close again
    file.close();
    CHECK(open_ids() == before);
    CHECK(file < 0);
    CHECK_NOTHROW(file.close());
  }
  CHECK(open_ids() == before);
}

TEST_CASE("HDF5 scope guard releases its identifier", "[hdf5]")
{
  const ssize_t before = open_ids();
  hid_t released = H5I_INVALID_HID;
  {
    hid_t handle
        = io::hdf5::open_file(MPI_COMM_SELF, test_file("handle"), "w", false);
    io::hdf5::Handle file(handle, H5Fclose);
    CHECK(open_ids() == before + 1);

    io::hdf5::Handle moved(std::move(file));
    CHECK(open_ids() == before + 1);
    released = moved.release();
  }

  // A released identifier is the caller's to close
  CHECK(open_ids() == before + 1);
  io::hdf5::close_file(released);
  CHECK(open_ids() == before);
}
