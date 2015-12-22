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

#ifdef HAS_HDF5

#include <string>
#include <vector>
#include <dolfin/mesh/CellType.h>

#include "pugixml.hpp"

namespace dolfin
{

  class XDMFxml
  {
  public:

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
    std::vector<std::string> topology_name() const;

    /// Get the (unique) Mesh geometry name, split into two parts
    /// (file name, dataset name) from the current XML
    std::vector<std::string> geometry_name() const;

    /// Get the (unique) dataset name for a MeshFunction in current
    /// XML
    std::string dataname() const;

    /// Add a data item to the current grid
    void data_attribute(std::string name,
                        std::size_t value_rank,
                        bool vertex_data,
                        std::size_t num_total_vertices,
                        std::size_t num_global_cells,
                        std::size_t padded_value_size,
                        std::string dataset_name);

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
                       const std::string reference);

    /// Attach geometry to the current grid node
    void mesh_geometry(const std::size_t num_total_vertices,
                       const std::size_t gdim,
                       const std::string reference);

  private:

    // Generate the XML header generic to all files
    void header();

    // The XML document
    pugi::xml_document xml_doc;

    // Current node for writing geometry, topology and data into
    pugi::xml_node xdmf_grid;

    // Filename
    std::string _filename;
  };
}
#endif
#endif
