// Copyright (C) 2012-2025 Chris N. Richardson, Jørgen S. Dokken
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include "HDF5Interface.h"
#include <array>
#include <basix/mdspan.hpp>
#include <boost/algorithm/string.hpp>
#include <boost/lexical_cast.hpp>
#include <complex>
#include <concepts>
#include <dolfinx/common/MPI.h>
#include <dolfinx/common/types.h>
#include <dolfinx/mesh/cell_types.h>
#include <filesystem>
#include <numeric>
#include <pugixml.hpp>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace dolfinx
{

namespace io::xdmf_utils
{

/// Get DOLFINx cell type string from XML topology node
/// @return DOLFINx cell type and polynomial degree
std::pair<std::string, int> get_cell_type(const pugi::xml_node& topology_node);

/// Return (0) HDF5 filename and (1) path in HDF5 file from a DataItem
/// node.
std::array<std::string, 2> get_hdf5_paths(const pugi::xml_node& dataitem_node);

std::filesystem::path
get_hdf5_filename(const std::filesystem::path& xdmf_filename);

/// Get dimensions from an XML DataSet node
std::vector<std::int64_t> get_dataset_shape(const pugi::xml_node& dataset_node);

/// Get number of cells from an XML Topology node
std::int64_t get_num_cells(const pugi::xml_node& topology_node);

/// Get the VTK string identifier
std::string vtk_cell_type_str(mesh::CellType cell_type, int num_nodes);

/// TODO: Document
template <typename T>
void add_data_item(pugi::xml_node& xml_node, hid_t h5_id,
                   const std::string& h5_path, std::span<const T> x,
                   std::int64_t offset, const std::vector<std::int64_t>& shape,
                   const std::string& number_type, bool use_mpi_io)
{
  // Add DataItem node
  assert(xml_node);
  pugi::xml_node data_item_node = xml_node.append_child("DataItem");
  assert(data_item_node);

  // Add dimensions attribute
  std::string dims;
  for (auto d : shape)
    dims += std::to_string(d) + std::string(" ");
  dims.pop_back();
  data_item_node.append_attribute("Dimensions") = dims.c_str();

  // Set type for topology data (needed by XDMF to prevent default to
  // float)
  if (!number_type.empty())
    data_item_node.append_attribute("NumberType") = number_type.c_str();

  // Add format attribute
  if (h5_id < 0)
  {
    data_item_node.append_attribute("Format") = "XML";
    assert(shape.size() == 2);
    std::ostringstream s;
    s.precision(16);
    for (std::size_t i = 0; i < x.size(); ++i)
    {
      if ((i + 1) % shape[1] == 0 and shape[1] != 0)
        s << x.data()[i] << std::endl;
      else
        s << x.data()[i] << " ";
    }

    data_item_node.append_child(pugi::node_pcdata).set_value(s.str().c_str());
  }
  else
  {
    data_item_node.append_attribute("Format") = "HDF";

    // Get name of HDF5 file, including path
    const std::filesystem::path p = io::hdf5::get_filename(h5_id);
    const std::filesystem::path filename = p.filename().c_str();

    // Add HDF5 filename and HDF5 internal path to XML file
    const std::string xdmf_path
        = filename.string() + std::string(":") + h5_path;
    data_item_node.append_child(pugi::node_pcdata).set_value(xdmf_path.c_str());

    // Compute data offset and range of values
    std::int64_t local_shape0 = std::reduce(
        std::next(shape.begin()), shape.end(), x.size(), std::divides{});

    const std::array local_range{offset, offset + local_shape0};
    io::hdf5::write_dataset(h5_id, h5_path, x.data(), local_range, shape,
                            use_mpi_io, false);

    // Add partitioning attribute to dataset
    // std::vector<std::size_t> partitions;
    // std::vector<std::size_t> offset_tmp(1, offset);
    // dolfinx::MPI::gather(comm, offset_tmp, partitions);
    // dolfinx::MPI::broadcast(comm, partitions);
    // io::hdf5::add_attribute(h5_id, h5_path, "partition", partitions);
  }
}

/// @brief Get data associated with a data set node.
/// @tparam T Data type to read into.
/// @warning Data will be silently cast to type `T` if requested type
/// and storage type differ.
template <typename T>
std::vector<T> get_dataset(MPI_Comm comm, const pugi::xml_node& dataset_node,
                           hid_t h5_id,
                           std::array<std::int64_t, 2> range = {0, 0})
{
  // FIXME: Need to sort out dataset dimensions - can't depend on HDF5
  // shape, and a Topology data item is not required to have a
  // 'Dimensions' attribute since the dimensions can be determined from
  // the number of cells and the cell type (for topology, one must
  // supply cell type + (number of cells or dimensions)).
  //
  // A geometry data item must have 'Dimensions' attribute.

  assert(dataset_node);
  pugi::xml_attribute format_attr = dataset_node.attribute("Format");
  assert(format_attr);

  // Get data set shape from 'Dimensions' attribute (empty if not
  // available)
  const std::vector shape_xml = xdmf_utils::get_dataset_shape(dataset_node);

  const std::string format = format_attr.as_string();
  std::vector<T> data_vector;
  // Only read ASCII on process 0
  const int mpi_rank = dolfinx::MPI::rank(comm);
  if (format == "XML")
  {
    if (mpi_rank == 0)
    {
      // Read data and trim any leading/trailing whitespace
      pugi::xml_node data_node = dataset_node.first_child();
      assert(data_node);
      std::string data_str = data_node.value();

      // Split data based on spaces and line breaks
      std::vector<boost::iterator_range<std::string::iterator>> data_vector_str;
      boost::split(data_vector_str, data_str, boost::is_any_of(" \n"));

      // Add data to numerical vector
      data_vector.reserve(data_vector_str.size());
      for (auto& v : data_vector_str)
      {
        if (v.begin() != v.end())
          data_vector.push_back(
              boost::lexical_cast<T>(boost::copy_range<std::string>(v)));
      }
    }
  }
  else if (format == "HDF")
  {
    // Get file and data path
    auto paths = xdmf_utils::get_hdf5_paths(dataset_node);

    // Get data shape from HDF5 file
    const std::vector shape_hdf5 = io::hdf5::get_dataset_shape(h5_id, paths[1]);

    // FIXME: should we support empty data sets?
    // Check that data set is not empty
    assert(!shape_hdf5.empty());
    assert(shape_hdf5[0] != 0);

    // Determine range of data to read from HDF5 file. This is
    // complicated by the XML Dimension attribute and the HDF5 storage
    // possibly having different shapes, e.g. the HDF5 storage may be a
    // flat array.

    // If range = {0, 0} then no range is supplied and we must determine
    // the range
    if (range[0] == 0 and range[1] == 0)
    {
      if (shape_xml == shape_hdf5)
      {
        range = dolfinx::MPI::local_range(mpi_rank, shape_hdf5[0],
                                          dolfinx::MPI::size(comm));
      }
      else if (!shape_xml.empty() and shape_hdf5.size() == 1)
      {
        // Size of dims > 0
        std::int64_t d = std::reduce(shape_xml.begin(), shape_xml.end(),
                                     std::int64_t(1), std::multiplies{});

        // Check for data size consistency
        if (d * shape_xml[0] != shape_hdf5[0])
        {
          throw std::runtime_error("Data size in XDMF/XML and size of HDF5 "
                                   "dataset are inconsistent");
        }

        // Compute data range to read
        range = dolfinx::MPI::local_range(mpi_rank, shape_xml[0],
                                          dolfinx::MPI::rank(comm));
        range[0] *= d;
        range[1] *= d;
      }
      else
      {
        throw std::runtime_error("This combination of array shapes in XDMF and "
                                 "HDF5 is not supported");
      }
    }

    // Retrieve data
    if (hid_t dset_id = io::hdf5::open_dataset(h5_id, paths[1]);
        dset_id == H5I_INVALID_HID)
      throw std::runtime_error("Failed to open HDF5 global dataset.");
    else
    {
      data_vector = io::hdf5::read_dataset<T>(dset_id, range, true);
      if (herr_t err = H5Dclose(dset_id); err < 0)
        throw std::runtime_error("Failed to close HDF5 global dataset.");
    }
  }
  else
    throw std::runtime_error("Storage format \"" + format + "\" is unknown");

  // Get dimensions for consistency (if available in DataItem node)
  if (shape_xml.empty())
  {
    std::int64_t size = 1;
    for (auto dim : shape_xml)
      size *= dim;

    std::int64_t size_global = 0;
    const std::int64_t size_local = data_vector.size();
    MPI_Allreduce(&size_local, &size_global, 1, MPI_INT64_T, MPI_SUM, comm);
    if (size != size_global)
    {
      throw std::runtime_error(
          "Data sizes in attribute and size of data read are inconsistent");
    }
  }

  return data_vector;
}

} // namespace io::xdmf_utils
} // namespace dolfinx
