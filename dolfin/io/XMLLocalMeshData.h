// Copyright (C) 2006 Skavhaug.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2008-11-28
// Last changed: 2008-11-28

#ifndef __XMLLOCALMESHDATA_H
#define __XMLLOCALMESHDATA_H

#include <dolfin/mesh/LocalMeshData.h>
#include "XMLObject.h"

/// Documentation of class XMLLocalMeshData

namespace dolfin
{
  class LocalMeshData;

  class XMLLocalMeshData: public XMLObject
  {
  public:
    
    /// Constructor
    XMLLocalMeshData(LocalMeshData& mesh_data);
    
    /// Destructor
    ~XMLLocalMeshData();
    
    void startElement (const xmlChar* name, const xmlChar** attrs);
    void endElement   (const xmlChar* name);
    
    void open(std::string filename);
    bool close();
    
  private:
    
    enum ParserState {OUTSIDE, INSIDE_DOLFIN, 
                      INSIDE_MESH, INSIDE_VERTICES, INSIDE_CELLS,
                      INSIDE_DATA, INSIDE_MESH_FUNCTION, INSIDE_ARRAY,
                      DONE};
    
    // Callbacks for reading XML data
    void readMesh        (const xmlChar* name, const xmlChar** attrs);
    void readVertices    (const xmlChar* name, const xmlChar** attrs);
    void readVertex      (const xmlChar* name, const xmlChar** attrs);
    void readCells       (const xmlChar* name, const xmlChar** attrs);
    void readInterval    (const xmlChar* name, const xmlChar** attrs);
    void readTriangle    (const xmlChar* name, const xmlChar** attrs);
    void readTetrahedron (const xmlChar* name, const xmlChar** attrs);
    void readMeshFunction(const xmlChar* name, const xmlChar** attrs);
    void readArray       (const xmlChar* name, const xmlChar** attrs);
    void readMeshEntity  (const xmlChar* name, const xmlChar** attrs);
    void readArrayElement(const xmlChar* name, const xmlChar** attrs);
    
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
