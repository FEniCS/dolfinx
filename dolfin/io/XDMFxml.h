// Copyright (C) 2015 Chris N. Richardson
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

#ifndef __DOLFIN_XDMFXML_H
#define __DOLFIN_XDMFXML_H

#include <array>
#include <string>
#include <vector>
#include <dolfin/mesh/CellType.h>

#include "pugixml.hpp"

namespace dolfin
{

  class XDMFxml
  {
  public:

    class TopologyData
    {
    public:
      std::string format;
      std::string hdf5_filename;
      std::string hdf5_dataset;
      std::string cell_type;
      std::size_t num_cells;
      std::size_t points_per_cell;
      std::string data;
    };

    class GeometryData
    {
    public:
      std::string format;
      std::string hdf5_filename;
      std::string hdf5_dataset;
      std::size_t num_points;
      std::size_t dim;
      std::string data;
    };

    /// Constructor
    XDMFxml(std::string filename);

    /// Destructor
    ~XDMFxml();

    /// Output to file
    void write() const;

    /// Read from a file
    void read();

    /// Get the (unique) Mesh topology name, split into three parts
    /// (file name, dataset name, CellType) from the current XML
    TopologyData get_topology() const;

    /// Get the (unique) Mesh geometry name, split into two parts
    /// (file name, dataset name) from the current XML
    GeometryData get_geometry() const;

    /// Get the (unique) dataset for a MeshFunction in current XML
    std::string get_first_data_set() const;

    /// Get the (unique) dataset name for a MeshFunction in current
    /// XML
    std::string dataname() const;

    /// Get the data encoding. "XML" or "HDF"
    std::string data_encoding() const;

    /// Add a data item to the current grid
    void data_attribute(std::string name,
                        std::size_t value_rank,
                        bool vertex_data,
                        std::size_t num_total_vertices,
                        std::size_t num_global_cells,
                        std::size_t padded_value_size,
                        std::string dataset_name,
                        std::string format);

    /// Initalise XML for a Mesh-like single output returning the
    /// xdmf_grid node
    pugi::xml_node init_mesh(std::string name);

    /// Initialise XML for a TimeSeries-like output returning the
    /// xdmf_grid node
    pugi::xml_node init_timeseries(std::string name, double time_step,
                                   std::size_t counter);

    /// Attach topology to the current grid node
    void mesh_topology(const CellType::Type cell_type,
                       const std::size_t cell_order,
                       const std::size_t num_global_cells,
                       const std::string xml_value_data,
                       const std::string format);

    /// Attach geometry to the current grid node
    void mesh_geometry(const std::size_t num_total_vertices,
                       const std::size_t gdim,
                       const std::string xml_value_data,
                       const std::string format,
                       const bool is_reference=false);

     // Split HDF5 paths (file path and internal HDF5 path)
     static
     std::array<std::string, 2> get_hdf5_paths(const pugi::xml_node& xml_node);

  private:

    // Generate the XML header generic to all files
    void header();


    // The XML document
    pugi::xml_document xml_doc;

    // Current node for writing geometry, topology and data into
    pugi::xml_node xdmf_grid;

    // Filename
    std::string _filename;

    // This is to ensure that when the file is written for the first
    // time, it overwrites any existing file with the same name.
    bool _is_this_first_write;
  };
}
#endif
