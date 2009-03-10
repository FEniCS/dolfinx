// Copyright (C) 2006 Skavhaug.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2009-03-10
// Last changed: 2009-03-10

#ifndef __NEWXMLLOCALMESHDATA_H
#define __NEWXMLLOCALMESHDATA_H

#include <dolfin/mesh/LocalMeshData.h>
#include "XMLHandler.h"

/// Documentation of class NewXMLLocalMeshData

namespace dolfin
{
  class LocalMeshData;

  class NewXMLLocalMeshData: public XMLHandler
  {
  public:
    
    /// Constructor
    NewXMLLocalMeshData(LocalMeshData& mesh_data, NewXMLFile& parser);
    
    /// Destructor
    ~NewXMLLocalMeshData();
    
    void start_element (const xmlChar* name, const xmlChar** attrs);
    void end_element   (const xmlChar* name);
    
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
    
    // Number of local vertices
    uint num_local_vertices() const;
    
    // Number of local cells
    uint num_local_cells() const;
    
    // Geometrical mesh dimesion
    uint gdim;
    
    // Topological mesh dimesion
    uint tdim;
        
    uint vertex_index_start, vertex_index_stop;
    uint cell_index_start, cell_index_stop;
    
    // Result object to build
    LocalMeshData& mesh_data;
    
    ParserState state;
    
  };

}
#endif
