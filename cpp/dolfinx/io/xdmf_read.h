// Copyright (C) 2012-2018 Chris N. Richardson and Garth N. Wells
//
// This file is part of DOLFINX (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include "pugixml.hpp"
#include "xdmf_utils.h"
#include <Eigen/Dense>
#include <boost/algorithm/string.hpp>
#include <boost/lexical_cast.hpp>
#include <dolfinx/common/IndexMap.h>
#include <dolfinx/mesh/Geometry.h>
#include <dolfinx/mesh/Mesh.h>
#include <dolfinx/mesh/Topology.h>

/// Low-level methods for reading XDMF files
namespace dolfinx::io::xdmf_read
{

/// Return data associated with a data set node
template <typename T>
std::vector<T> get_dataset(MPI_Comm comm, const pugi::xml_node& dataset_node,
                           const hid_t h5_id,
                           std::array<std::int64_t, 2> range = {{0, 0}})
{
  // FIXME: Need to sort out datasset dimensions - can't depend on HDF5
  // shape, and a Topology data item is not required to have a
  // 'Dimensions' attribute since the dimensions can be determined from
  // the number of cells and the cell type (for topology, one must
  // supply cell type + (number of cells or dimensions).
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
    const std::vector shape_hdf5
        = HDF5Interface::get_dataset_shape(h5_id, paths[1]);

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
        std::int64_t d = std::accumulate(shape_xml.begin(), shape_xml.end(), 1,
                                         std::multiplies<std::int64_t>());

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
        throw std::runtime_error(
            "This combination of array shapes in XDMF and HDF5 "
            "is not supported");
      }
    }

    // Retrieve data
    data_vector = HDF5Interface::read_dataset<T>(h5_id, paths[1], range);
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
//----------------------------------------------------------------------------

} // namespace dolfinx::io::xdmf_read
