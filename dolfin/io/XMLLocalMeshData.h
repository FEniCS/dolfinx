// Copyright (C) 2006 Ola Skavhaug
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
// First added:  2009-03-10
// Last changed: 2011-05-30
// Modified by Kent-Andre Mardal, 2011.

#ifndef __XMLLOCALMESHDATA_H
#define __XMLLOCALMESHDATA_H

#include <boost/scoped_ptr.hpp>
#include <dolfin/mesh/LocalMeshData.h>
#include "OldXMLFile.h"
#include "XMLHandler.h"

namespace dolfin
{

  class LocalMeshData;
  class XMLArray;

  /// Documentation of class XMLLocalMeshData

  class XMLLocalMeshData: public XMLHandler
  {
  public:

    /// Constructor
    XMLLocalMeshData(LocalMeshData& mesh_data, OldXMLFile& parser);

    /// Destructor
    ~XMLLocalMeshData();

    void start_element(const xmlChar* name, const xmlChar** attrs);
    void end_element(const xmlChar* name);

  private:

    enum ParserState {OUTSIDE,
                      INSIDE_MESH, INSIDE_VERTICES, INSIDE_CELLS,
                      INSIDE_DATA, INSIDE_MESH_FUNCTION, INSIDE_ARRAY,
                      INSIDE_DATA_ENTRY,
                      DONE};

    // Callbacks for reading XML data
    void read_mesh        (const xmlChar* name, const xmlChar** attrs);
    void read_vertices    (const xmlChar* name, const xmlChar** attrs);
    void read_vertex      (const xmlChar* name, const xmlChar** attrs);
    void read_cells       (const xmlChar* name, const xmlChar** attrs);
    void read_interval    (const xmlChar* name, const xmlChar** attrs);
    void read_triangle    (const xmlChar* name, const xmlChar** attrs);
    void read_tetrahedron (const xmlChar* name, const xmlChar** attrs);
    void read_mesh_function(const xmlChar* name, const xmlChar** attrs);
    void read_array        (const xmlChar* name, const xmlChar** attrs);
    void read_mesh_data    (const xmlChar* name, const xmlChar** attrs);
    void read_data_entry   (const xmlChar* name, const xmlChar** attrs);

    // Number of local vertices
    uint num_local_vertices() const;

    // Number of local cells
    uint num_local_cells() const;

    // Geometrical mesh dimesion
    uint gdim;

    // Topological mesh dimesion
    uint tdim;

    // Range for vertices
    std::pair<uint, uint> vertex_range;

    // Range for cells
    std::pair<uint, uint> cell_range;

    // Result object to build
    LocalMeshData& mesh_data;

    // Name of the array
    std::string data_entry_name;

    // State of parser
    ParserState state;

    // Use for reading embedded array data
    boost::scoped_ptr<XMLArray> xml_array;

  };

}
#endif
