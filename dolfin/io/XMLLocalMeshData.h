// Copyright (C) 2006 Ola Skavhaug.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2009-03-10
// Last changed: 2009-09-08

#ifndef __XMLLOCALMESHDATA_H
#define __XMLLOCALMESHDATA_H

#include <dolfin/mesh/LocalMeshData.h>
#include "XMLHandler.h"


namespace dolfin
{
  class LocalMeshData;

  /// Documentation of class XMLLocalMeshData

  class XMLLocalMeshData: public XMLHandler
  {
  public:

    /// Constructor
    XMLLocalMeshData(LocalMeshData& mesh_data, XMLFile& parser);

    /// Destructor
    ~XMLLocalMeshData();

    void start_element(const xmlChar* name, const xmlChar** attrs);
    void end_element(const xmlChar* name);

  private:

    enum ParserState {OUTSIDE,
                      INSIDE_MESH, INSIDE_VERTICES, INSIDE_CELLS,
                      INSIDE_DATA, INSIDE_MESH_FUNCTION, INSIDE_ARRAY,
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

    ParserState state;

  };

}
#endif
